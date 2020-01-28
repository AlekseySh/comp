import random
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple, List, DefaultDict, Counter as TCounter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from fastai.basic_train import Learner
from fastai.tabular import *
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.nn.functional import softmax
from tqdm.auto import tqdm

TStat = Dict[
    Tuple[str, ...],  # (2018_stat_day_month, ...)
    TCounter[Tuple[str, ...]]  # {('Sid421", 'Friday'): 25, ...}
]

TFieldComb = List[Tuple[str, ...]]


def add_velocity(data: pd.DataFrame, velocity_path: Path) -> None:
    velocity = pd.read_csv(velocity_path, parse_dates=['time'])

    velocity['time'] = velocity.time.dt.floor('H')

    dsid_str = 'datetime x segment_id'
    velocity[dsid_str] = velocity.time.astype(str) + ' x ' + velocity.sid

    dsid_to_vehcount = defaultdict(lambda: np.nan, zip(velocity[dsid_str], velocity.veh_count))
    dsid_to_vel = defaultdict(lambda: np.nan, zip(velocity[dsid_str], velocity.vel))

    data['vel'] = data[dsid_str].apply(lambda dsid: dsid_to_vel[dsid])
    data['veh_count'] = data[dsid_str].apply(lambda dsid: dsid_to_vehcount[dsid])

    print('Velocity data was added.')


def add_zeros(data_ones: pd.DataFrame) -> pd.DataFrame:
    trange = pd.date_range('2016-01-01', '2019-04-01', freq='1h')[:-1]

    sids = list(data_ones.sid.unique())

    ttrange = trange.repeat(len(sids))
    ssids = pd.Series(sids * len(trange))

    data = pd.DataFrame({'datetime x segment_id': ttrange.astype(str) + ' x ' + ssids,
                         'time': pd.to_datetime(ttrange), 'sid': ssids})

    data = data.join(data_ones.set_index('datetime x segment_id'),
                     on='datetime x segment_id',
                     how='left', rsuffix='_ones')

    data = data.drop(columns=['time_ones', 'sid_ones'])
    data = data.fillna(value=0)

    assert ((min(data.time) <= data_ones.time) & (data_ones.time <= max(data.time))).all()
    assert sum(data.target) == len(data_ones)

    print('Zeros were added.')
    return data


def read_vehinj(vehinj_dir: Path) -> pd.DataFrame:
    veh = pd.read_csv(vehinj_dir / 'Vehicles2016_2019.csv',
                      parse_dates=['CreatedLOcalDateTime'],
                      usecols=['CreatedLOcalDateTime', 'EventID'],
                      dayfirst=True
                      )

    inj = pd.read_csv(vehinj_dir / 'Injuries2016_2019.csv',
                      parse_dates=['Created Local Date Time'],
                      usecols=['Created Local Date Time', 'Event Id'],
                      dayfirst=True
                      )

    veh = veh.rename(columns={'CreatedLOcalDateTime': 'time'})
    inj = inj.rename(columns={'Created Local Date Time': 'time',
                              'Event Id': 'EventID'})

    vehinj = pd.concat([veh, inj], sort=False).dropna()  # only 1 rec is nat
    vehinj['EventID'] = vehinj['EventID'].astype(int)

    return vehinj


def add_vehinj(data: pd.DataFrame, vehinj_dir: Path) -> None:
    vehinj = read_vehinj(vehinj_dir)

    vehinj['time'] = vehinj.time.dt.floor('H')

    vehinj_counter = Counter(vehinj.time)

    data['n_vehinj'] = data.time.apply(lambda tkey: vehinj_counter[tkey])

    print('Veh and inj data were added.')


def add_sid_data(data: pd.DataFrame, road_segments: pd.DataFrame) -> None:
    assert 'sid' in data.columns.values.tolist()

    def sid_to_smth(default_val: Any,
                    field_name: str
                    ) -> DefaultDict[str, Any]:
        return defaultdict(lambda: default_val,
                           zip(road_segments.segment_id, road_segments[field_name])
                           )

    road_segments.WIDTH = road_segments.WIDTH.astype('str')
    road_segments.LANES = road_segments.LANES.astype('str')

    sids = np.array(data.sid)

    sid_to_length = sid_to_smth(500, 'length_1')
    sid_to_condition = sid_to_smth('Good', 'CONDITION')
    sid_to_width = sid_to_smth('0.0', 'WIDTH')
    sid_to_roadno = sid_to_smth('N1', 'ROADNO')
    sid_to_lanes = sid_to_smth('2', 'LANES')

    length_1 = np.zeros(len(data))
    condition = np.chararray(len(data), 10)
    width = np.chararray(len(data), 5)
    roadno = np.chararray(len(data), 15)
    lanes = np.chararray(len(data), 5)

    for i, sid in tqdm(enumerate(sids), total=len(sids)):
        length_1[i] = sid_to_length[sid]
        condition[i] = str(sid_to_condition[sid])
        width[i] = str(sid_to_width[sid])
        roadno[i] = str(sid_to_roadno[sid])
        lanes[i] = str(sid_to_lanes[sid])

    data['length_1'] = length_1
    data['condition'] = condition
    data['width'] = width
    data['roadno'] = roadno
    data['lanes'] = lanes

    print('Sid data was added.')


def add_more_time(data: pd.DataFrame) -> None:
    pd.options.mode.chained_assignment = None
    assert 'time' in data.columns

    data['hour'] = data.time.dt.hour
    data['day'] = data.time.dt.day_name()
    data['month'] = data.time.dt.month_name()

    data['day_n'] = data.time.dt.day
    data['month_n'] = data.time.dt.month

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


def split_low_high(df: pd.DataFrame,
                   n_most_low: int,
                   verbose: bool = True
                   ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Drop sids with smallest number of events

    sid_counter = Counter(df.sid)
    sorted_counts = sorted(list(sid_counter.values()))
    th = sorted_counts[n_most_low]
    sids_low = set([sid for sid, count in sid_counter.items() if count < th])
    sids_high = set(df.sid) - sids_low

    assert sids_high.symmetric_difference(sids_low) == set(df.sid)

    w_high = df.sid.isin(sids_high)
    df_high = df[w_high]
    df_low = df[~w_high]

    df_low.reset_index(drop=True, inplace=True)
    df_high.reset_index(drop=True, inplace=True)

    if verbose:
        ratio = 1 - sum(sorted_counts[n_most_low:]) / sum(sorted_counts)
        print(f'We select {n_most_low} sids, it is'
              f' {round(100 * ratio, 3)}% of'
              f' events, th: {th} events per history.')

    return df_high, df_low


def select_by_time(df: pd.DataFrame,
                   tstart: str,
                   tend: str,
                   time_col: str = 'time'
                   ) -> pd.DataFrame:
    df_select = df[(pd.Timestamp(tstart) <= df[time_col]) &
                   (df[time_col] < pd.Timestamp(tend))]

    df_select.reset_index(drop=True, inplace=True)

    return df_select


def check_fields(field_combs: TFieldComb, data: pd.DataFrame) -> None:
    fields_required = set([field for comb in field_combs for field in comb])
    fields = set(data.columns.values.tolist())

    assert fields_required.issubset(fields), (fields_required, fields)


def calculate_statistic(data_stat: pd.DataFrame,
                        field_combs: TFieldComb
                        ) -> TStat:
    check_fields(field_combs, data_stat)

    stat: TStat = {comb: Counter() for comb in field_combs}

    for row in data_stat.itertuples():

        for comb in field_combs:
            field_values = [getattr(row, field) for field in comb]

            stat[comb][tuple(field_values)] += 1

    return stat


def add_statistic(data: pd.DataFrame,
                  stat_data: pd.DataFrame,
                  tstart: str,
                  tend: str,
                  field_combs: TFieldComb,
                  prefix: str
                  ) -> pd.DataFrame:
    check_fields(field_combs, data)

    stat_data = select_by_time(stat_data, tstart, tend)

    stat = calculate_statistic(stat_data, field_combs)

    vectors = {comb: np.zeros(len(data)) for comb in field_combs}

    print('Statistic appending:')
    for row in tqdm(data.itertuples(), total=len(data)):

        for comb in field_combs:
            field_values = [getattr(row, field) for field in comb]

            vectors[comb][row.Index] = stat[comb][tuple(field_values)]

    period_coef = (pd.Timestamp(tend) - pd.Timestamp(tstart)).days
    for key in vectors.keys():
        data[prefix + '_'.join(key)] = vectors[key] / period_coef

    print('Statistic data was added added.')


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
             th_start=0, th_stop=1, steps=20
             ):
    if isinstance(model, Learner):
        probas_tensor, *_ = model.get_preds(DatasetType.Valid)
        probas = probas_tensor[:, 1].cpu().numpy()
        y_true = y_true
    else:
        probas = np.array(model.predict_proba(validation_pool))[:, 1]
        y_true = np.array(validation_pool.get_label())

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


def f1_flexible(probas, gts, th_start, th_stop, steps):
    th_grid = np.linspace(start=th_start, stop=th_stop, num=steps)

    scs = [f1_score(y_pred=probas > th, y_true=gts) for th in th_grid]

    i_max = np.argmax(scs)

    return scs[i_max], th_grid[i_max]


def f1_flexible_clip(probas, gts, th_start, th_stop, steps, th_score):
    f1_max, th_max = f1_flexible(probas, gts, th_start, th_stop, steps)

    if f1_max < th_score:
        th_max = 1

    return f1_max, th_max


def select_th_for_each_sid(data: pd.DataFrame,
                           probas: np.ndarray,
                           th_score: float
                           ) -> Dict[str, float]:
    sids_list = data.segment_id.unique()
    sid_to_th = {}

    for sid in tqdm(sids_list):
        w_sid = (data.segment_id == sid).values

        _, th_max = f1_flexible_clip(probas=np.array(probas[:, 1])[w_sid],
                                     gts=np.array(data['y'])[w_sid],
                                     th_start=0, th_stop=1, steps=20,
                                     th_score=th_score
                                     )
        sid_to_th[sid] = th_max

    return sid_to_th


def make_predict_many_th(data: pd.DataFrame,
                         sid_to_th: Dict[str, float],
                         probas: np.ndarray
                         ) -> np.ndarray:
    assert len(data) == len(probas)

    sids_list = data.segment_id.unique()

    predict = np.zeros(len(data))

    for sid in sids_list:
        w_sid = data.segment_id == sid
        predict[w_sid] = probas[w_sid] > sid_to_th[sid]

    return predict


def run_many_th_experiment(data: pd.DataFrame, probas: np.ndarray, th_score: float) -> None:
    n = len(data)

    assert n == len(probas)

    i_split = n // 2 - 10
    ii_shuffle = np.random.permutation(n)
    ii_train, ii_val = ii_shuffle[:i_split], ii_shuffle[i_split:]

    train = data.loc[ii_train]
    val = data.loc[ii_val]

    sid_to_th = select_th_for_each_sid(train, probas[ii_train], th_score=th_score)

    predict_train = make_predict_many_th(train, sid_to_th, probas[ii_train, 1])
    predict_val = make_predict_many_th(val, sid_to_th, probas[ii_val, 1])

    f1_train = f1_score(y_true=data.y[ii_train], y_pred=predict_train)
    f1_val = f1_score(y_true=data.y[ii_val], y_pred=predict_val)

    print(f'New score train: {f1_train}, new scre val: {f1_val}.')


class F1(Callback):

    def __init__(self, th_start=0, th_stop=1, steps=20):
        self.th_start = th_start
        self.th_stop = th_stop
        self.steps = steps

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
