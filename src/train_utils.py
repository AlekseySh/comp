import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from fastai.basic_data import DatasetType
from fastai.basic_train import Learner
from fastai.callback import Callback
from fastai.tabular import add_metrics
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from torch.nn.functional import softmax
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')


def select_by_time(df: pd.DataFrame,
                   tstart: str,
                   tend: str,
                   time_col: str = 'time'
                   ) -> pd.DataFrame:
    df_select = df[(pd.Timestamp(tstart) <= df[time_col]) &
                   (df[time_col] < pd.Timestamp(tend))]

    df_select.reset_index(drop=True, inplace=True)

    return df_select


def estimate(model, validation_pool, y_true,
             th_start=0, th_stop=1, steps=101, pred_probas=None
             ):
    if pred_probas is None:

        if isinstance(model, Learner):
            probas_tensor, *_ = model.get_preds(DatasetType.Valid)
            probas = probas_tensor[:, 1].cpu().numpy()
            y_true = y_true
        else:
            probas = np.array(model.predict_proba(validation_pool))[:, 1]
            y_true = np.array(validation_pool.get_label())

    else:
        probas = pred_probas

    ths = np.linspace(th_start, th_stop, steps)

    sc = [f1_score(y_true=y_true, y_pred=probas > th) for th in tqdm(ths)]

    i_max = np.argmax(sc)
    score_max, th_max = sc[i_max], ths[i_max]
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

    plt.plot(ths, sc)
    plt.grid('on')
    plt.show()

    return th_max


def random_seed(seed_value: int = 42) -> None:
    use_cuda = torch.cuda.is_available()
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed


# ======== Visualization ========

def plot_events(events_time: pd.Series) -> None:
    t_start = min(events_time)
    n_days = (max(events_time) - t_start).days

    events_img = np.zeros((24, n_days + 1), np.int8)

    for t in events_time:
        events_img[t.hour, (t - t_start).days] = 1

    assert np.sum(events_img) == len(set(events_time.dt.floor('H')))

    yy = np.arange(0, 24, 1)
    xx = (np.arange(0, n_days + 1, 1))

    plt.figure(figsize=(16, 16))
    plt.imshow(events_img)
    plt.yticks(yy - .5, [str(y) for y in yy])
    plt.xticks(xx + .5, [str(x) for x in xx])
    plt.grid(True)
    plt.show()


def plot_sid_events(data: pd.DataFrame, sid: str, tstart: str, tend: str) -> None:
    data_sid = data[data.sid == sid]
    data_sid = select_by_time(data_sid, tstart, tend)

    print(sid, ':', tstart, '-', tend, f'{len(data_sid.time)} events')
    plot_events(data_sid.time)


# ======== Utils for models ========


def f1_flexible(probas, gts, th_start, th_stop, steps):
    th_grid = np.linspace(start=th_start, stop=th_stop, num=steps)
    scs = [f1_score(y_pred=probas > th, y_true=gts) for th in th_grid]
    id_max = np.argmax(scs)
    return scs[id_max], th_grid[id_max]


class F1(Callback):
    # Callback for Fastai neural model

    def __init__(self, th_start=0, th_stop=1, steps=20):
        self.th_start = th_start
        self.th_stop = th_stop
        self.steps = steps
        self.probas, self.gts = [], []

    def on_epoch_begin(self, **kwargs):
        self.probas, self.gts = [], []

    def on_batch_end(self, last_output, last_target, **kwargs):
        probas = softmax(last_output, dim=1)[:, 1].cpu().numpy()
        self.probas.extend(probas)

        self.gts.extend(last_target.cpu().tolist())

    def on_epoch_end(self, last_metrics, **kwargs):
        m, _ = f1_flexible(probas=np.array(self.probas),
                           gts=np.array(self.gts),
                           th_start=self.th_start,
                           th_stop=self.th_stop,
                           steps=self.steps
                           )
        return add_metrics(last_metrics, m)


class FlexibleF1(object):
    # F1 calculator for CatBoost

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

            f1_scores = [f1_score(y_pred=probs > th,
                                  y_true=np.array(target)
                                  )
                         for th in self.th_grid]

            score = max(f1_scores)

            return score, 1.0
