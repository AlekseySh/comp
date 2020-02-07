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


def vote_predict(probas, ths):
    assert probas.shape[0] == len(ths)

    n_model = len(ths)
    counts = np.zeros(probas.shape[1])
    for i_model in range(n_model):
        pred_i = probas[i_model, :] > ths[i_model]

        counts += pred_i

    pred = counts >= np.ceil(n_model / 2)
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


def check():
    train_data, train_labels, eval_data, eval_labels, eval_dataset, cat_features = get_data()

    train_pool = Pool(data=train_data, label=train_labels, cat_features=cat_features)
    val_pool = Pool(data=eval_data, label=eval_labels, cat_features=cat_features)

    init_params = {'learning_rate': 0.1,
                   'task_type': 'CPU',
                   'has_time': True,
                   'eval_metric': 'F1'
                   }

    train_params = {}

    model = train(init_params=init_params,
                  train_params=train_params,
                  n_iter=200,
                  train_pool=train_pool,
                  val_pool=val_pool,
                  eval_freq=10,
                  stopper_n_obs=30,
                  stopper_delta=0.01
                  )

    print(model)


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


def get_data():
    cat_features = [0, 1, 2]

    train_data = [["a", "b", 1, 4, 5, 6],
                  ["a", "b", 4, 5, 6, 7],
                  ["c", "d", 30, 40, 50, 60]]

    train_labels = [1, 1, 0]

    eval_data = [["a", "c", 3, 4, 4, 1],
                 ["a", "d", 1, 5, 5, 5],
                 ["b", "d", 31, 25, 60, 70],
                 ["b", "a", 31, 1, 60, 70],
                 ["b", "a", 31, 1, 2, 1]]

    eval_labels = [0, 1, 1, 0, 1]

    eval_dataset = Pool(eval_data,
                        label=eval_labels,
                        cat_features=cat_features)

    return train_data, train_labels, eval_data, eval_labels, eval_dataset, cat_features


def check_flexible():
    train_data, train_labels, eval_data, eval_labels, eval_dataset, cat_features = get_data()

    # Initialize CatBoostClassifier with custom `eval_metric`
    flexible_f1 = FlexibleF1(0.1, 0.7, 10)
    model = CatBoostClassifier(iterations=5,
                               eval_metric=flexible_f1,
                               loss_function='Logloss')

    # Fit model with `use_best_model=True`
    model.fit(train_data,
              train_labels,
              cat_features,
              use_best_model=True,
              eval_set=eval_dataset)

    # Get predictions
    pred = model.predict_proba(eval_data)

    scores = [f1_score(y_pred=pred[:, 1] > th,
                       y_true=np.array(eval_labels)
                       )
              for th in flexible_f1.th_grid]

    score = max(scores)
    print('score', score)


def check_th():
    train_data, train_labels, eval_data, eval_labels, eval_dataset, cat_features = get_data()

    # Initialize CatBoostClassifier with custom `eval_metric`
    th_f1 = FlexibleF1(0.5, 0.5, 1)
    n_it = 5
    model = CatBoostClassifier(iterations=n_it,
                               eval_metric=th_f1,
                               loss_function='Logloss')

    # Fit model with `use_best_model=True`
    model.fit(train_data,
              train_labels,
              cat_features,
              use_best_model=True,
              eval_set=eval_dataset)

    # lib version
    model2 = CatBoostClassifier(iterations=n_it,
                                eval_metric='F1',
                                loss_function='Logloss')

    # Fit model with `use_best_model=True`
    model2.fit(train_data,
               train_labels,
               cat_features,
               use_best_model=True,
               eval_set=eval_dataset)


if __name__ == '__main__':
    check_flexible()
    check_th()
    check()
