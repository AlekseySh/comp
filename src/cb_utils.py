import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score

import src.utils as u

warnings.filterwarnings('ignore')


def fit_predict_cv(train: pd.DataFrame,
                   test: pd.DataFrame,
                   val_ids: List[np.ndarray],
                   cat_cols: List[str],
                   all_cols: List[str],
                   draw_plot: bool = True
                   ) -> Tuple[np.ndarray, List[float], List[float]]:
    u.random_seed(42)

    assert len(set(test.datetime).intersection(set(train.datetime))) == 0

    params = {
        'iterations': 1000,
        'learning_rate': 0.5,
        'scale_pos_weight': 10,
        'has_time': False,
        'one_hot_max_size': 1000,
        'depth': 8,

        'loss_function': 'Logloss',
        'task_type': 'CPU',
        'use_best_model': True,
        'eval_metric': FlexibleF1(0, 1, 20)
    }

    exp_dir = Path('/home/AlekseySh/code/comp/results/cb/' + str(datetime.now()))
    exp_dir.mkdir(exist_ok=True, parents=True)

    main_pool = Pool(data=train[all_cols], label=train['y'], cat_features=cat_cols)
    test_pool = Pool(data=test[all_cols], cat_features=cat_cols)

    probas = np.zeros((len(val_ids), len(test)), dtype=np.float16)
    val_scores, ths = [], []
    for i, cur_val_ids in enumerate(val_ids):
        print(f'Fold {i + 1} / {len(val_ids)} started.')

        cur_train_ids = np.setdiff1d(np.arange(len(train)),
                                     np.squeeze(cur_val_ids))

        train_pool = main_pool.slice(cur_train_ids)
        val_pool = main_pool.slice(cur_val_ids)

        cls = CatBoostClassifier(**params)

        cls.fit(
            train_pool,
            eval_set=val_pool,
            plot=True,
            verbose=draw_plot,
            early_stopping_rounds=30
        )

        cls.save_model(str(exp_dir / f'model_{i}.pt'))

        score, th = u.f1_flexible(probas=cls.predict_proba(val_pool)[:, 1],
                                  gts=val_pool.get_label(), th_start=0,
                                  th_stop=1, steps=20)
        ths.append(th)

        probas[i, :] = cls.predict_proba(test_pool)[:, 1]
        val_scores.append(score)

    print(f'CV mean score is {np.mean(val_scores)}.')

    return probas, ths, val_scores


def load_models(path_to_ckpt):
    models = []
    for nm in list(Path(path_to_ckpt).glob('**/model*.pt')):
        cls = CatBoostClassifier()
        models.append(cls.load_model(str(path_to_ckpt / nm)))
    return models


def predict_multi_models(models, ths, pool, mode='vote'):
    n_model = len(models)
    probas = np.zeros((n_model, pool.num_row()))
    for i in range(n_model):
        probas[i, :] = models[i].predict_proba(pool)[:, 1]

    if mode == 'vote':
        pred = m.vote_predict(probas, ths)

    else:
        raise Exception

    return pred


def train_resume_gpu(init_params,
                     train_params,
                     n_iter,
                     train_pool,
                     val_pool,
                     eval_freq,
                     stopper_n_obs,
                     stopper_delta,
                     ):
    ths = np.linspace(start=0, stop=1, num=20)

    stopper = Stopper(n_obs=np.ceil(stopper_n_obs / eval_freq),
                      delta=stopper_delta
                      )

    n_step = int(np.ceil(n_iter / eval_freq))

    cur_model = CatBoostClassifier(iterations=eval_freq, **init_params)
    cur_model.fit(train_pool, **train_params)

    score = flexible_f1(model=cur_model, pool=val_pool, ths=ths)
    stopper.update(cur_val=score)
    print(0, score)

    for i_step in range(1, n_step + 1):
        model = CatBoostClassifier(iterations=eval_freq, **init_params)
        model.fit(train_pool, init_model=cur_model, **train_params)
        cur_model = model
        del model

        score = flexible_f1(model=cur_model, pool=val_pool, ths=ths)
        stopper.update(cur_val=score)

        print(i_step, score)

        if stopper.check_criterion():
            break

    return cur_model


class FlexibleF1(object):

    def __init__(self,
                 th_start: float = 0.0,
                 th_stop: float = 1.0,
                 steps: int = 20
                 ):
        self.th_grid = np.linspace(start=th_start,
                                   stop=th_stop,
                                   num=steps
                                   )
        self.train_call = False

    @staticmethod
    def is_max_optimal() -> bool:
        return True

    @staticmethod
    def get_final_error(error, _):
        return error

    def evaluate(self, approxes, target, _) -> float:
        self.train_call = ~self.train_call

        if self.train_call:
            return 0, 1.0

        else:
            assert len(approxes) == 1
            assert len(target) == len(approxes[0])

            approx = np.array(approxes[0])

            exps = np.exp(approx)
            probs = exps / (1 + exps)

            scores = [f1_score(y_pred=probs > th,
                               y_true=np.array(target)
                               )
                      for th in self.th_grid]

            score = max(scores)

            return score, 1.0
