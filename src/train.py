from functools import partial
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from fastai.callbacks import EarlyStoppingCallback, SaveModelCallback
from fastai.tabular import FillMissing, Categorify, Normalize, TabularList, tabular_learner
from fastai.train import ShowGraph
from matplotlib import pyplot as plt
from torch import tensor
from torch.nn import CrossEntropyLoss as CEloss

from src.train_utils import random_seed, F1


def read_data(train_path: Path,
              test_path: Path
              ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str], List[str]]:
    test = pd.read_pickle(test_path)

    random_seed(42)
    train = test
    train['y'] = np.random.rand(len(train)) > 0.5
    train['y'] = train['y'].astype(int)
    # train = pd.read_pickle(train_path)

    cols_to_drop = ['datetime x segment_id', 'datetime',
                    'lane_width', 'y', 'main_route', 'speed_unknown']

    all_cols = list(set(train.columns.values) - set(cols_to_drop))

    cat_cols = ['segment_id', 'weekday', 'month', 'hour',
                'ROADNO', 'CLASS', 'LANES', 'SURFTYPE',
                'PAVETYPE', 'CONDITION', 'vds_id', 'wind_dir',
                'weather_cond', 'cloud_1', 'cloud_2',
                'cloud_3', 'cloud_cover_fog', 'wind_dir_defined',
                'mist', 'fog', 'smoke', 'rain', 'drizzle', 'snow',
                'traffic_unknown', 'public_holiday', 'school_holiday',
                'day_period', 'average_ttime_na']

    cont_cols = list(set(all_cols) - set(cat_cols))

    test[cat_cols] = test[cat_cols].replace(np.nan, 'NAN').astype(str)
    train[cat_cols] = train[cat_cols].replace(np.nan, 'NAN').astype(str)

    test['y'].replace(np.nan, '', inplace=True)

    for f in cont_cols:
        test[f] = test[f].fillna(0)
        train[f] = train[f].fillna(0)

    return train, test, all_cols, cont_cols, cat_cols


def create_fai_databunch(train: pd.DataFrame,
                         test: pd.DataFrame,
                         cat_cols: List[str],
                         cont_cols: List[str],
                         seed: int
                         ):
    # val_ids = train[train.datetime >= pd.Timestamp('2018-10-01')].index
    val_ids = (range(0, 100_000))

    procs = [FillMissing, Categorify, Normalize]

    random_seed(seed)

    test_tab = TabularList.from_df(df=test, cat_names=cat_cols, cont_names=cont_cols)

    random_seed(seed)
    data = (TabularList.from_df(
        train, procs=procs, cat_names=cat_cols, cont_names=cont_cols)
            .split_by_idx(val_ids)
            .label_from_df(cols='y')
            .add_test(test_tab)
            .databunch(bs=10_000, num_workers=1)
            )

    return data


def train_fai_model(data, seed: int):
    random_seed(seed)
    learn = tabular_learner(data, layers=[1024, 512, 256, 128],
                            metrics=F1(th_start=0, th_stop=1, steps=101),
                            callback_fns=[ShowGraph,
                                          partial(EarlyStoppingCallback,
                                                  monitor='f1',
                                                  min_delta=0.0001,
                                                  patience=4)
                                          ],
                            loss_func=CEloss(
                                weight=tensor([1, 10]).float().cuda()
                            ),
                            opt_func=torch.optim.Adam
                            )

    learn.lr_find()
    learn.recorder.plot()
    plt.show()

    random_seed(seed)
    learn.fit_one_cycle(3, max_lr=slice(5e-3),
                        callbacks=[SaveModelCallback(learn, every='improvement',
                                                     monitor='f1', name='best_epoch')]
                        )

    learn.recorder.plot_losses()
    learn.recorder.plot_lr()
    plt.show()

    return learn


def main():
    train_path = '../data/train_1002.pkl'
    test_path = '../data/test_1002.pkl'
    seed = 42

    train, test, all_cols, cont_cols, cat_cols = read_data(train_path=train_path, test_path=test_path)

    databunch = create_fai_databunch(train=train, test=test,
                                     cat_cols=cat_cols, cont_cols=cont_cols, seed=seed)

    learn = train_fai_model(data=databunch, seed=seed)


if __name__ == '__main__':
    main()
