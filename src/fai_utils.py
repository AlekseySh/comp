from typing import List, Tuple

import numpy as np
from fastai.basic_train import Learner, Recorder


def fit_predict_cv(val_ids: List[np.ndarray],
                   learn: Learner,
                   n_epochs: int,
                   lr: slice
                   ) -> Tuple[np.ndarray, List[float]]:
    assert learn.data.empty_val
    assert isinstance(val_ids, List)

    k_fold = len(val_ids)

    pred_probas_arr = np.zeros((k_fold, len(learn.data.test_ds)),
                               dtype=np.float16)

    val_scores = []
    for i_fold, cur_val_ids in enumerate(val_ids):
        print(f'Fold {1 + i_fold} / {k_fold} start.')

        learn.data.split_by_idx(cur_val_ids)

        learn.lr_find()

        learn.fit_one_cycle(n_epochs, lr,
                            callbacks=[
                                SaveModelCallback(learn, every='improvement',
                                                  monitor='f1', name=f'best{i_fold}')
                            ])

        probas, *_ = learn.get_preds(DatasetType.Test)
        pred_probas_arr[i_fold, :] = probas

        score = get_max_from_log(learn.recorder)
        val_scores.append(score)

    return pred_probas_arr, val_scores


def get_max_from_log(recorder: Recorder,
                     metric_name: str = 'f1'
                     ) -> float:
    i_metric = recorder.metrics_names.index(metric_name)
    metrics_log = recorder.metrics[i_metric]

    i_max = np.argmax(metrics_log)
    max_score = metrics_log[i_max]

    print(f'Best score arised on {i_max} epoch: {max_score}.')

    return max_score
