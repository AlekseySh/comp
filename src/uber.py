import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

UBER_DATA_PATH = Path(__file__).parent.parent / 'data' / 'uber'


def make_uzone_to_enum(enum_to_uzone):
    uzone_to_enum = {}

    for routno, enum_to_uzone_rout in enum_to_uzone.items():

        uzone_to_enum_rout = {}

        for enum, zones in enum_to_uzone[routno].items():

            if "None" in zones:
                zones.remove("None")

            for z in zones:
                uzone_to_enum_rout[z] = int(enum)

        uzone_to_enum[routno] = uzone_to_enum_rout

    return uzone_to_enum


def load_uber_data():
    def load(fname):
        return json.load(open(UBER_DATA_PATH / fname, 'r'))

    routes = load('routes.json')
    sid_to_enum = load('sid_to_enum.json')
    enum_to_uzone = load('enum_to_uzone.json')

    uzones_times = pd.read_csv(UBER_DATA_PATH / 'uzones_times.csv',
                               parse_dates=['Date']
                               )
    uzones_times = uzones_times.rename(columns={
        'Origin Display Name': 'zone_a',
        'Destination Display Name': 'zone_b'
    })

    return routes, sid_to_enum, enum_to_uzone, uzones_times


def assign_ttimes_to_sids(routes, sid_to_enum, uzone_to_enum, zone_ab_to_time):
    sids, enums = list(zip(*sid_to_enum.items()))
    sid_times = pd.DataFrame(data={'segment_id': sids, 'enums': enums})
    sid_times['ttime_to_center'] = np.nan
    sid_times['ttime_from_center'] = np.nan

    for routno, route in routes.items():
        assert len(route['to_center']) == len(route['from_center'])

        route_len = len(route['to_center'])

        sid_times_arr = np.zeros((2, route_len), dtype=np.float16)  # 0 - to_center, 1 - from center
        n_hits = np.zeros_like(sid_times_arr)

        for (zone_a, zone_b), travel_time in zone_ab_to_time.items():

            if travel_time is None:
                continue

            enum_a = uzone_to_enum[routno][zone_a]
            enum_b = uzone_to_enum[routno][zone_b]

            is_from_center = route['from_center'].index(enum_a) < route['from_center'].index(enum_b)
            direction, ax = ('from_center', 1) if is_from_center else ('to_center', 0)

            i_a, i_b = route[direction].index(enum_a), route[direction].index(enum_b)

            sid_times_arr[ax, i_a: i_b + 1] += travel_time / (i_b - i_a + 1)
            n_hits[ax, i_a: i_b + 1] += 1

        sid_times_arr = sid_times_arr / n_hits

        enum_to_ttive_to_center = dict(zip(route['to_center'], sid_times_arr[0]))
        enum_to_ttive_from_center = dict(zip(route['from_center'], sid_times_arr[1]))

        w_route_to = sid_times.enums.isin(route['to_center'])
        sid_times['ttime_to_center'].loc[w_route_to] = \
            sid_times.enums.loc[w_route_to].apply(lambda enum: enum_to_ttive_to_center[enum])

        w_route_from = sid_times.enums.isin(route['from_center'])
        sid_times['ttime_from_center'].loc[w_route_from] = \
            sid_times.enums.loc[w_route_from].apply(lambda enum: enum_to_ttive_from_center[enum])

    return sid_times


def main():
    routes, sid_to_enum, enum_to_uzone, ut = load_uber_data()
    uzone_to_enum = make_uzone_to_enum(enum_to_uzone)

    dates = pd.date_range('2016-01-01', '2019-03-31', freq='1d')

    day_period = ['AM', 'PM', 'Midday', 'Early Morning', 'Evening']

    for date in tqdm(dates):
        ut_cur = ut[ut.Date == pd.Timestamp(date)]
        ut_cur.reset_index(inplace=True)

        for period in day_period:
            zone_ab_to_time = dict(zip(
                list(zip(ut_cur.zone_a, ut_cur.zone_b)),
                ut_cur[period + ' Mean Travel Time (Seconds)'])
            )

            sids_times = assign_ttimes_to_sids(routes=routes,
                                               sid_to_enum=sid_to_enum,
                                               uzone_to_enum=uzone_to_enum,
                                               zone_ab_to_time=zone_ab_to_time
                                               )
            sids_times['date'] = date
            sids_times['day_period'] = period


if __name__ == '__main__':
    main()
