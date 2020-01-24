import warnings

import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score

from src.model_utils import get_data

warnings.filterwarnings('ignore')


def flexible_f1(model, pool, ths):
    probs = model.predict_proba(pool)[:, 1]

    scores = [f1_score(y_pred=probs > th,
                       y_true=pool.get_label()
                       )
              for th in ths]

    i_max = np.argmax(scores)

    return scores[i_max], ths[i_max]


def train(init_params,
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

    score, _ = flexible_f1(model=cur_model, pool=val_pool, ths=ths)
    stopper.update(cur_val=score)
    print(0, score)

    for i_step in range(1, n_step + 1):
        model = CatBoostClassifier(iterations=eval_freq, **init_params)
        model.fit(train_pool, init_model=cur_model, **train_params)
        cur_model = model
        del model

        score, _ = flexible_f1(model=cur_model, pool=val_pool, ths=ths)
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


if __name__ == '__main__':
    check()
