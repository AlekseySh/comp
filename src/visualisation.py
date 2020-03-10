from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def select_by_time(df: pd.DataFrame,
                   tstart: str,
                   tend: str,
                   time_col: str = 'time'
                   ) -> pd.DataFrame:
    df_select = df[(pd.Timestamp(tstart) <= df[time_col]) &
                   (df[time_col] < pd.Timestamp(tend))]

    df_select.reset_index(drop=True, inplace=True)

    return df_select


def plot_sid_events(data: pd.DataFrame, sid: str, tstart: str, tend: str) -> None:
    data_sid = data[data.sid == sid]
    data_sid = select_by_time(data_sid, tstart, tend)

    print(sid, ':', tstart, '-', tend, f'{len(data_sid.time)} events')
    plot_events(data_sid.time)


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
