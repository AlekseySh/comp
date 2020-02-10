import random
import warnings
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from fastai.basic_train import Learner
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')


def read_data(data_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str], List[str]]:
    train = pd.read_pickle(data_path / 'train_0802_fin.pkl')
    test = pd.read_pickle(data_path / 'test_0802_final.pkl')

    cols_to_drop = ['datetime x segment_id',
                    'datetime',
                    'acc_count_sid_hour',
                    'acc_count_sid_weekday',
                    'acc_count_sid_month',
                    'acc_count_vds_hour',
                    'acc_count_vds_weekday',
                    'acc_count_vds_month',
                    'y']

    all_cols = list(set(train.columns.values) - set(cols_to_drop))

    cat_cols = ['segment_id', 'weekday', 'month', 'hour',
                'ROADNO', 'CLASS', 'LANES', 'SURFTYPE',
                'PAVETYPE', 'CONDITION', 'vds_id', 'wind_dir',
                'weather_cond', 'cloud_1', 'cloud_2',
                'cloud_3', 'cloud_cover_fog', 'wind_dir_defined',
                'mist', 'fog', 'smoke', 'rain', 'drizzle', 'snow',
                'traffic_unknown', 'speed_unknown',
                'public_holiday', 'school_holiday', 'day_period',
                'average_ttime_na']

    cont_cols = list(set(all_cols) - set(cat_cols))

    test[cat_cols] = test[cat_cols].replace(np.nan, 'NAN').astype(str)
    train[cat_cols] = train[cat_cols].replace(np.nan, 'NAN').astype(str)

    return train, test, all_cols, cont_cols, cat_cols


def add_more_time(data: pd.DataFrame) -> None:
    pd.options.mode.chained_assignment = None
    assert 'time' in data.columns

    data['hour'] = data.time.dt.hour
    data['day'] = data.time.dt.day_name()
    data['month'] = data.time.dt.month_name()

    data['day_n'] = data.time.dt.day
    data['month_n'] = data.time.dt.month

    data['weekday'] = data['time'].dt.weekday

    print('Time data was added.')


def read_ones(train_path: Path) -> pd.DataFrame:
    bad_sids = {
        '-33.8891283413',
        '-33.9622761744',
        '-33.9680008638',
        '-34.0436786939',
        '-34.0894652753'
    }

    ones = pd.read_csv(
        train_path,
        parse_dates=['Occurrence Local Date Time'],
        usecols=['Occurrence Local Date Time', 'road_segment_id']
    )

    ones = ones.rename(columns={
        'Occurrence Local Date Time': 'time',
        'road_segment_id': 'sid'}
    )

    ones['time'] = ones.time.dt.floor('H')
    ones['target'] = 1

    ones['datetime x segment_id'] = ones.time.astype(str) + ' x ' + ones.sid
    ones = ones.drop_duplicates('datetime x segment_id')
    ones = ones[~ones.sid.isin(bad_sids)]
    ones = ones.sort_values(by='time')

    ones.reset_index(drop=True, inplace=True)

    return ones


def select_by_time(df: pd.DataFrame,
                   tstart: str,
                   tend: str,
                   time_col: str = 'time'
                   ) -> pd.DataFrame:
    df_select = df[(pd.Timestamp(tstart) <= df[time_col]) &
                   (df[time_col] < pd.Timestamp(tend))]

    df_select.reset_index(drop=True, inplace=True)

    return df_select


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


def scores(y_true: np.array,
           y_pred: np.array,
           verbose: bool = True
           ) -> Tuple[float, float, float]:
    f = f1_score(y_true=y_true, y_pred=y_pred)
    r = recall_score(y_true=y_true, y_pred=y_pred)
    p = precision_score(y_true=y_true, y_pred=y_pred)

    if verbose:
        print(f'f1: {f}, recall: {r}, precision: {p}')

    return f, r, p


def estimate(model, validation_pool, y_true,
             th_start=0, th_stop=1, steps=20, pred_probas=None
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


def proc_silent_intervals(data: pd.DataFrame) -> pd.DataFrame:
    silent_ints = list(map(
        lambda x: (pd.Timestamp(x[0]), pd.Timestamp(x[1])),
        [
            ('2016-07-14 17:00', '2016-08-01 07:00'),
            ('2016-08-14 17:00', '2016-09-01 08:00'),
            ('2018-08-13 16:00', '2018-09-01 06:00'),
        ]
    ))

    exception_sids = {'03RHJ3G', '0ICKV72', '0PU7VDI',
                      '0W39BFY', '16WNX7T', '1K4ZYII',
                      '2J6C2D5', '3IQ1GWG', '8LOVJZ3',
                      '8PK91S2', 'AJRKP0C', 'BC5XKSB',
                      'CTB99FS', 'D7SS5LM', 'DRNRL0M',
                      'F5UCVMI', 'H9QJECU', 'IUTMY1U',
                      'JT4HGZ2', 'K3N8ADC', 'L4CWZBU',
                      'M4U0X5G', 'MRQ81XJ', 'Q03FQ74',
                      'Q0VL8BD', 'SQ3W7J8', 'TC7A716',
                      'TONSERE', 'UUZT4OE', 'UXERJVK',
                      'VBUCV9N', 'W0EUG1C', 'WRJK3P3',
                      'XOCWI97', 'YGRV6SD', 'YNCIDMW'}
    # this sids are still active even in "silence" periods

    is_silent = data.datetime.apply(
        lambda t0: any([(a < t0) & (t0 < b) for (a, b) in silent_ints])
    )

    w_left = data.segment_id.isin(exception_sids) | (~is_silent)
    data_res = data[w_left]
    data_res.reset_index(drop=True, inplace=True)

    n, n_res = len(data), len(data_res)
    print(f'{n - n_res} records inside the silents periods were dropped.')

    assert sum(data_res.y.astype(bool)) == sum(data.y.astype(bool))

    return data_res


def random_seed(seed_value: int = 42) -> None:
    use_cuda = torch.cuda.is_available()
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


# === Utils for models ===

class F1(Callback):
    # Callback for fastai neural model

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
    # F1 calculator for catboost

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
