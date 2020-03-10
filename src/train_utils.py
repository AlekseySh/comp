import datetime
import random
import warnings
from functools import partial
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import torch
from fastai.basic_data import DatasetType
from fastai.basic_train import Learner
from fastai.callback import Callback
from fastai.callbacks import EarlyStoppingCallback, SaveModelCallback
from fastai.tabular import (
    add_metrics, FillMissing, Categorify, Normalize,
    TabularList, tabular_learner, DataBunch
)
from fastai.train import ShowGraph
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from torch import tensor
from torch.nn import CrossEntropyLoss as CEloss
from torch.nn.functional import softmax

warnings.filterwarnings('ignore')


def estimate(learn: Learner,
             th_start: float = 0.0,
             th_stop: float = 1.0,
             steps: int = 20
             ) -> Tuple[float, float]:
    probas_tensor, *_ = learn.get_preds(DatasetType.Valid)
    probas = probas_tensor[:, 1].cpu().numpy()

    y_true = learn.data.label_list.valid.y.items

    score_max, th_max = f1_flexible(probas=probas, gts=y_true, th_start=th_start,
                                    th_stop=th_stop, steps=steps)

    events = sum(probas > th_max)

    precision_max = precision_score(y_true=y_true, y_pred=probas > th_max)
    recall_max = recall_score(y_true=y_true, y_pred=probas > th_max)

    print('f1 score ', round(score_max, 4),
          '\n recall', round(recall_max, 4),
          '\n precision', round(precision_max, 4),
          '\n th ', round(th_max, 4),
          '\n events  pred', events,
          '\n events true', sum(y_true)
          )

    return th_max, score_max


def random_seed(seed_value: int = 42) -> None:
    use_cuda = torch.cuda.is_available()
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed


def f1_flexible(probas: np.ndarray,
                gts: np.ndarray,
                th_start: float,
                th_stop: float,
                steps: int,
                need_plot: bool = False
                ) -> Tuple[float, float]:
    th_grid = np.linspace(start=th_start, stop=th_stop, num=steps)
    scs = [f1_score(y_pred=probas > th, y_true=gts) for th in th_grid]
    id_max = np.argmax(scs)

    if need_plot:
        plt.plot(th_grid, scs)
        plt.grid('on')
        plt.show()

    return scs[id_max], th_grid[id_max]


class F1(Callback):
    # Callback for Fastai neural model

    def __init__(self, th_start: float = 0.0, th_stop: float = 1.0, steps: int = 20):
        self.th_start = th_start
        self.th_stop = th_stop
        self.steps = steps

        self.probas: List[float] = []
        self.gts: List[float] = []

    def on_epoch_begin(self, **kwargs: Dict[str, Any]) -> None:
        self.probas, self.gts = [], []

    def on_batch_end(self,
                     last_output: torch.Tensor,
                     last_target: torch.Tensor,
                     **kwargs: Dict[str, Any]
                     ) -> None:
        probas = softmax(last_output, dim=1)[:, 1].cpu().numpy()
        self.probas.extend(probas)

        self.gts.extend(last_target.cpu().tolist())

    def on_epoch_end(self, last_metrics: Any, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        m, _ = f1_flexible(probas=np.array(self.probas),
                           gts=np.array(self.gts),
                           th_start=self.th_start,
                           th_stop=self.th_stop,
                           steps=self.steps
                           )
        return add_metrics(last_metrics, m)


def read_data(train_path: Path,
              test_path: Path
              ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str], List[str]]:
    test = pd.read_pickle(test_path)
    train = pd.read_pickle(train_path)

    cols_to_drop = ['datetime x segment_id', 'datetime',
                    'lane_width', 'y', 'main_route', 'speed_unknown',
                    'mean_average_ttime', 'std_average_ttime']

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
        if 'next' in f.lower():
            test[f] = test[f].fillna(0)
            train[f] = train[f].fillna(0)

    return train, test, all_cols, cont_cols, cat_cols


def create_fai_databunch(train: pd.DataFrame,
                         test: pd.DataFrame,
                         cat_cols: List[str],
                         cont_cols: List[str],
                         val_start: str
                         ) -> DataBunch:
    # train = train[train.datetime >= '2016-10-01']
    # train.reset_index(inplace=True, drop=True)

    val_ids = train[train.datetime >= pd.Timestamp(val_start)].index

    procs = [FillMissing, Categorify, Normalize]

    test_tab = TabularList.from_df(df=test, cat_names=cat_cols, cont_names=cont_cols)

    data = (TabularList.from_df(
        train, procs=procs, cat_names=cat_cols, cont_names=cont_cols)
            .split_by_idx(val_ids)
            .label_from_df(cols='y')
            .add_test(test_tab)
            .databunch(bs=100_000, num_workers=4)
            )

    return data


def train_fai_model(data: DataBunch) -> Learner:
    time = str(datetime.datetime.now().time())

    work_dir = Path(f'../results/{time}')
    work_dir.mkdir(exist_ok=True, parents=True)

    learn = tabular_learner(data, path=work_dir, layers=[1024, 512, 256, 128],
                            metrics=F1(th_start=0, th_stop=1, steps=51),
                            callback_fns=[ShowGraph,
                                          partial(EarlyStoppingCallback,
                                                  monitor='f1',
                                                  min_delta=0.00001,
                                                  patience=3)
                                          ],
                            loss_func=CEloss(
                                weight=tensor([1, 10]).float().cuda()
                            ))
    learn.lr_find()
    lr_slice = slice(5e-3)  # type: ignore
    learn.fit_one_cycle(10, max_lr=lr_slice,
                        callbacks=[SaveModelCallback(learn, every='improvement',
                                                     monitor='f1', name='best_model')]
                        )

    return learn
