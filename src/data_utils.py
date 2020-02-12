import datetime
import math
from typing import Dict, Tuple, Optional, Iterable, List

import numpy as np
import pandas as pd
from pysolar.solar import get_altitude, get_azimuth
from pytz import timezone
from shapely.geometry import LineString

CAPETOWN_CENTER_LAT = -34.270836
CAPETOWN_CENTER_LON = 18.459778

CAPETOWN_TIMEZONE = timezone('Africa/Johannesburg')

PUBLIC_HOLIDAYS = ['2016-01-01', '2016-03-21', '2016-03-25', '2016-03-28',
                   '2016-04-27', '2016-05-01', '2016-05-02', '2016-06-16',
                   '2016-08-03', '2016-08-09', '2016-09-24', '2016-12-16',
                   '2016-12-25', '2016-12-26', '2016-12-27',

                   '2017-01-01', '2017-01-02', '2017-03-21', '2017-04-14',
                   '2017-04-17', '2017-04-27', '2017-05-01', '2017-06-16',
                   '2017-08-09', '2017-09-24', '2017-09-25', '2017-12-16',
                   '2017-12-25', '2017-12-26',

                   '2018-01-01', '2018-03-21', '2018-03-30', '2018-04-02',
                   '2018-04-27', '2018-05-01', '2018-06-16', '2018-08-09',
                   '2018-09-24', '2018-12-16', '2018-12-17', '2018-12-25',
                   '2018-12-26',

                   '2019-01-01', '2019-03-21']

SCHOOL_HOLIDAYS_BE = ['2016-01-01', '2016-01-12',
                      '2016-03-19', '2016-04-04',
                      '2016-06-25', '2016-07-17',
                      '2016-10-01', '2016-10-09',
                      '2016-12-08', '2016-12-31',

                      '2017-01-01', '2017-01-10',
                      '2017-04-01', '2017-04-17',
                      '2017-07-01', '2017-07-23',
                      '2017-09-30', '2017-10-08',
                      '2017-12-07', '2017-12-31',

                      '2018-01-01', '2018-01-16',
                      '2018-03-29', '2018-04-09',
                      '2018-06-23', '2018-07-16',
                      '2018-09-29', '2018-10-08',
                      '2018-12-13', '2018-12-31',

                      '2019-01-01', '2019-01-08',
                      '2019-03-16', '2019-03-31']


def optimize_floats(df: pd.DataFrame) -> pd.DataFrame:
    floats = df.select_dtypes(include=['float64']).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast='float')
    return df


def optimize_ints(df: pd.DataFrame) -> pd.DataFrame:
    ints = df.select_dtypes(include=['int64']).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast='integer')
    return df


def optimize(df: pd.DataFrame) -> pd.DataFrame:
    return optimize_floats(optimize_ints(df))


def get_sinuosity(s: LineString) -> float:
    return s.length / math.sqrt((s.coords[0][0] - s.coords[-1][0]) ** 2 +
                                (s.coords[0][1] - s.coords[-1][1]) ** 2)


def get_alt(t: pd.Timestamp) -> float:
    t = t.replace(tzinfo=CAPETOWN_TIMEZONE)
    return get_altitude(CAPETOWN_CENTER_LAT, CAPETOWN_CENTER_LON, t)


def get_az(t: pd.Timestamp) -> float:
    t = t.replace(tzinfo=CAPETOWN_TIMEZONE)
    return get_azimuth(CAPETOWN_CENTER_LAT, CAPETOWN_CENTER_LON, t)


def get_orientation(s: LineString) -> float:
    degrees = math.degrees(math.atan2((s.coords[-1][0] - s.coords[0][0]),
                                      (s.coords[-1][1] - s.coords[0][1])))
    if degrees < 0:
        degrees += 360

    return degrees


def add_lag_features_vds(df: pd.DataFrame) -> pd.DataFrame:
    periods = df['vds_id'].nunique()
    for c in ['avg_speed', 'traffic_total', 'rel_diff_avg_speed', 'rel_diff_traffic']:
        assert (df['vds_id'][df.index % periods == 0].nunique() == 1)
        df['delta_' + c + '_last_hour'] = df[c].diff(periods=periods)
        df['delta_' + c + '_next_hour'] = -df[c].diff(periods=-periods)
    return df


def add_lag_features_uber(df: pd.DataFrame) -> pd.DataFrame:
    periods = df['segment_id'].nunique()
    for c in ['average_ttime']:
        assert (df['segment_id'][df.index % periods == 0].nunique() == 1)
        df['delta_' + c + '_last_day'] = df[c].diff(periods=periods)
        df['delta_' + c + '_last_week'] = df[c].diff(periods=7 * periods)
        df['delta_' + c + '_next_day'] = -df[c].diff(periods=-periods)
        df['delta_' + c + '_next_week'] = -df[c].diff(periods=-7 * periods)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['weekday'] = df['datetime'].dt.weekday
    df['month'] = df['datetime'].dt.month
    df['hour'] = df['datetime'].dt.hour
    df['date'] = df['datetime'].dt.date.astype(str)
    df['s_hour'] = round(np.sin(2 * np.pi * df['hour'] / 24), 3)
    df['c_hour'] = round(np.cos(2 * np.pi * df['hour'] / 24), 3)
    df['s_month'] = round(np.sin(2 * np.pi * df['month'] / 12), 3)
    df['c_month'] = round(np.cos(2 * np.pi * df['month'] / 12), 3)
    return df


def add_feature_based_on_neighbs(data: pd.DataFrame,
                                 fval: str,
                                 num_to_neighbors: Dict[str, List[Optional[int]]]
                                 ) -> None:
    siddate_to_val = dict(zip(zip(data['num'], data['datetime']), data[fval]))

    def get_val(n_dt: Tuple[str, pd.Timestamp]) -> Optional[float]:
        num, dt = n_dt
        neighbs_vals = [siddate_to_val.get((n, dt), np.nan) for n in num_to_neighbors[num]]
        avg_val = np.nanmean(neighbs_vals)
        return avg_val

    data[fval + '_neigh'] = list(map(lambda n_dt: get_val(n_dt),
                                     zip(data['num'], data['datetime']))
                                 )


def add_public_holidays(df: pd.DataFrame) -> pd.DataFrame:
    public_holidays = [datetime.datetime.strptime(date, '%Y-%m-%d').date() for date in PUBLIC_HOLIDAYS]

    df['public_holiday'] = df.datetime.dt.date.isin(public_holidays).replace(0, np.nan)
    df['public_holiday'] = df['public_holiday'].fillna(method='bfill', axis=0, limit=2 * 24 * 544)
    df['public_holiday'] = df['public_holiday'].fillna(method='ffill', axis=0, limit=2 * 24 * 544)
    df['public_holiday'] = df['public_holiday'].fillna(0).astype(int)
    return df


def daterange(date1: pd.Timestamp, date2: pd.Timestamp) -> Iterable[pd.Timestamp]:
    for n in range(int((date2 - date1).days) + 1):
        yield date1 + datetime.timedelta(n)


def add_school_holidays(df: pd.DataFrame) -> pd.DataFrame:
    school_holiday_be = [datetime.datetime.strptime(date, '%Y-%m-%d').date() for date in SCHOOL_HOLIDAYS_BE]
    school_holiday = []

    for d in range(len(school_holiday_be) // 2):
        for dt in daterange(school_holiday_be[2 * d], school_holiday_be[2 * d + 1]):
            school_holiday.append(dt)

    df['school_holiday'] = df['datetime'].dt.date.isin(school_holiday).astype(int)
    return df


def proc_silent_intervals(data: pd.DataFrame) -> pd.DataFrame:
    # We found intervals without accidents for majority part
    # of sids except sids which listed in `exception_sids`

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


def clean_vds_data(vds_hourly: pd.DataFrame,
                   vds_locations: pd.DataFrame
                   ) -> pd.DataFrame:
    vds_hourly['Site name'] = vds_hourly['Site name'].replace({'DS ': '',
                                                               'V': 'VDS',
                                                               'WTRNX ': 'VDS',
                                                               ' IB': 'I',
                                                               ' OB': 'O',
                                                               ' South': 'S',
                                                               ' North': 'N',
                                                               ' ': ''}, regex=True)

    vds_id_to_edit1 = vds_hourly.loc[~vds_hourly['Site name'].isin(vds_locations['Asset Desc.'])]['Site name'].unique()

    vds_id_edited1 = [el[:-1] for el in vds_id_to_edit1]

    vds_hourly['Site name'] = vds_hourly['Site name'].replace(vds_id_to_edit1, vds_id_edited1)

    vds_id_to_edit2 = vds_hourly.loc[~vds_hourly['Site name'].isin(vds_locations['Asset Desc.'])]['Site name'].unique()

    vds_id_edited2 = ['VDS123', 'VDS220A', 'VDS269', 'VDS269', 'VDS405A', 'VDS407',
                      'VDS705', 'VDS705', 'VDS266', 'VDS266', 'VDS113', 'VDS112', 'VDS112']

    vds_hourly['Site name'] = vds_hourly['Site name'].replace(vds_id_to_edit2, vds_id_edited2)

    vds_hourly['Site name'] = vds_hourly['Site name'].replace({'VDS139I': 'VDS139',
                                                               'VDS139O': 'VDS139',
                                                               'VDS912I': 'VDS912',
                                                               'VDS912O': 'VDS912',
                                                               'VDS913I': 'VDS913',
                                                               'VDS913O': 'VDS913',
                                                               'VDS914I': 'VDS914',
                                                               'VDS914O': 'VDS914',
                                                               'VDS135O': 'VDS135I',
                                                               'VDS130': 'VDS132O',
                                                               'VDS309': 'VDS308',
                                                               'VDS404I': 'VDS404O',
                                                               'VDS700I': 'VDS700O'}, regex=True)

    vds_hourly['Date of Collection Period'] += pd.to_timedelta(vds_hourly['Hour of Collection Period'], unit='h')

    vds_hourly = vds_hourly.drop('Hour of Collection Period', axis=1)
    vds_hourly.columns = ['vds_id', 'datetime', 'avg_speed', 'traffic1', 'traffic2', 'traffic3']
    vds_hourly['traffic_total'] = vds_hourly['traffic1'] + vds_hourly['traffic2'] + vds_hourly['traffic3']

    vds_hourly = vds_hourly.groupby(['vds_id', 'datetime']).agg({'traffic1': 'mean',
                                                                 'traffic2': 'mean',
                                                                 'traffic3': 'mean',
                                                                 'traffic_total': 'mean',
                                                                 'avg_speed': 'mean'
                                                                 }).reset_index()
    return vds_hourly
