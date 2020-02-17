import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

TAnyDict = Dict[str, Any]

TEnumToUzone = Dict[str, Dict[str, List[str]]]


def make_uzone_to_enum(enum_to_uzone: TEnumToUzone
                       ) -> Dict[str, Dict[str, int]]:
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


def load_uber_data(routes_path: str,
                   sid_to_enum_path: str,
                   enum_to_uzone_path: str,
                   routes_length_path: str,
                   uzone_times_path: str
                   ) -> Tuple[TAnyDict,
                              Dict[str, int],
                              TEnumToUzone,
                              pd.DataFrame,
                              TAnyDict]:
    def load(fpath: str) -> TAnyDict:
        return json.load(open(fpath, 'r'))

    routes = load(routes_path)
    sid_to_enum = load(sid_to_enum_path)
    enum_to_uzone = load(enum_to_uzone_path)
    routes_length = load(routes_length_path)

    uzones_times = pd.read_csv(uzone_times_path, parse_dates=['Date'])
    uzones_times = uzones_times.rename(columns={
        'Origin Display Name': 'zone_a',
        'Destination Display Name': 'zone_b'
    })

    return routes, sid_to_enum, enum_to_uzone, uzones_times, routes_length


def find(l, x):  # type: ignore
    if x in l:
        return l.index(x)
    else:
        return None


def assign_ttimes_to_sids(routes: TAnyDict,
                          routes_length: TAnyDict,
                          sid_to_enum: Dict[str, int],
                          uzone_to_enum: TAnyDict,
                          zone_ab_to_time: Dict[Tuple[str, str], float]
                          ) -> pd.DataFrame:
    sids, enums = list(zip(*sid_to_enum.items()))
    sid_times = pd.DataFrame(data={'segment_id': sids, 'enums': enums})
    sid_times['ttime_to_center'] = np.nan
    sid_times['ttime_from_center'] = np.nan

    for routno, route in routes.items():

        route_len_to_center = len(route['to_center'])
        route_len_from_center = len(route['from_center'])

        length_to_center = np.array(routes_length[routno]['to_center'])
        length_from_center = np.array(routes_length[routno]['from_center'])

        sid_times_to_center = np.zeros(route_len_to_center, dtype=np.float16)
        sid_times_from_center = np.zeros(route_len_from_center, dtype=np.float16)

        n_hits_to_center = np.zeros_like(sid_times_to_center)
        n_hits_from_center = np.zeros_like(sid_times_from_center)

        for (zone_a, zone_b), travel_time in zone_ab_to_time.items():

            if travel_time is None:
                continue

            enum_a = uzone_to_enum[routno].get(zone_a, None)
            enum_b = uzone_to_enum[routno].get(zone_b, None)

            if None in (enum_a, enum_b):
                continue

            idx_a_from = find(route['from_center'], enum_a)
            idx_b_from = find(route['from_center'], enum_b)

            idx_a_to = find(route['to_center'], enum_a)
            idx_b_to = find(route['to_center'], enum_b)

            if (None in (idx_a_from, idx_b_from)) & (None in (idx_a_to, idx_b_to)):
                continue

            if (None not in (idx_a_from, idx_b_from)) & (None in (idx_a_to, idx_b_to)):
                if idx_a_from < idx_b_from:
                    is_from_center = True
                else:
                    continue

            if (None in (idx_a_from, idx_b_from)) & (None not in (idx_a_to, idx_b_to)):
                if idx_a_to < idx_b_to:
                    is_from_center = False
                else:
                    continue

            if (None not in (idx_a_from, idx_b_from)) & (None not in (idx_a_to, idx_b_to)):
                if idx_a_from < idx_b_from:
                    is_from_center = True
                else:
                    is_from_center = False

            direction = 'from_center' if is_from_center else 'to_center'

            i_a, i_b = route[direction].index(enum_a), route[direction].index(enum_b)
            ab = slice(i_a, i_b + 1)

            if is_from_center:
                sid_times_from_center[ab] += travel_time * (length_from_center[ab] / sum(length_from_center[ab]))
                n_hits_from_center[ab] += 1
            else:
                sid_times_to_center[ab] += travel_time * (length_to_center[ab] / sum(length_to_center[ab]))
                n_hits_to_center[ab] += 1

        sid_times_to_center = sid_times_to_center / n_hits_to_center
        sid_times_from_center = sid_times_from_center / n_hits_from_center

        enum_to_ttive_to_center = dict(zip(route['to_center'], sid_times_to_center))
        enum_to_ttive_from_center = dict(zip(route['from_center'], sid_times_from_center))

        w_route_to = sid_times.enums.isin(route['to_center'])
        sid_times['ttime_to_center'].loc[w_route_to] = \
            sid_times.enums.loc[w_route_to].apply(lambda enum: enum_to_ttive_to_center[enum])

        w_route_from = sid_times.enums.isin(route['from_center'])
        sid_times['ttime_from_center'].loc[w_route_from] = \
            sid_times.enums.loc[w_route_from].apply(lambda enum: enum_to_ttive_from_center[enum])

    return sid_times


def process_uber_data(routes_path: str,
                      sid_to_enum_path: str,
                      enum_to_uzone_path: str,
                      routes_length_path: str,
                      uzone_times_path: str,
                      result_segment_ttime_path: str
                      ) -> None:
    if Path(result_segment_ttime_path).is_file():
        print(f'File {result_segment_ttime_path} is already exists.')
        return

    routes, sid_to_enum, enum_to_uzone, ut, routes_length = load_uber_data(
        routes_path=routes_path, sid_to_enum_path=sid_to_enum_path,
        enum_to_uzone_path=enum_to_uzone_path, routes_length_path=routes_length_path,
        uzone_times_path=uzone_times_path
    )
    uzone_to_enum = make_uzone_to_enum(enum_to_uzone)

    dates = pd.date_range('2016-01-01', '2019-03-31', freq='1d')

    day_period = ['Daily']  # ['AM', 'PM', 'Midday', 'Early Morning', 'Evening']

    segment_id_times = pd.DataFrame()

    for date in tqdm(dates):
        ut_cur = ut[ut.Date == pd.Timestamp(date)]
        ut_cur.reset_index(inplace=True)

        for period in day_period:
            zone_ab_to_time = dict(zip(
                list(zip(ut_cur.zone_a, ut_cur.zone_b)),
                ut_cur[period + ' Mean Travel Time (Seconds)'])
            )
            sids_times = assign_ttimes_to_sids(routes=routes,
                                               routes_length=routes_length,
                                               sid_to_enum=sid_to_enum,
                                               uzone_to_enum=uzone_to_enum,
                                               zone_ab_to_time=zone_ab_to_time)

            sids_times['date'] = date
            sids_times['day_period'] = period

            segment_id_times = pd.concat([segment_id_times, sids_times])

    # Postprocessing
    segment_id_times.reset_index(drop=True, inplace=True)

    segment_id_times.ttime_to_center.fillna(segment_id_times.ttime_from_center, inplace=True)
    segment_id_times.ttime_from_center.fillna(segment_id_times.ttime_to_center, inplace=True)

    segment_id_times['average_ttime'] = (segment_id_times.ttime_to_center + segment_id_times.ttime_from_center) / 2
    segment_id_times['average_ttime_na'] = segment_id_times['average_ttime'].isnull().astype(int)

    segment_id_times['year'] = segment_id_times.date.dt.year
    segment_id_times['month'] = segment_id_times.date.dt.month
    segment_id_times['weekday'] = segment_id_times.date.dt.weekday

    segment_id_times.average_ttime.fillna(
        segment_id_times.groupby(['segment_id', 'year', 'month', 'weekday'])['average_ttime'].transform('mean'),
        inplace=True)

    segment_id_times.average_ttime.fillna(
        segment_id_times.groupby(['segment_id', 'month', 'weekday'])['average_ttime'].transform('mean'),
        inplace=True)

    segment_id_times.average_ttime.fillna(
        segment_id_times.groupby(['segment_id', 'weekday'])['average_ttime'].transform('mean'),
        inplace=True)

    segment_id_times[['date', 'segment_id', 'average_ttime', 'average_ttime_na']].to_csv(
        result_segment_ttime_path, index=False)


def concat_uber_files(uber_files_dir: Path, result_fpath: str) -> None:
    if Path(result_fpath).is_file():
        print(f'File {result_fpath} is already exists.')
        return

    all_files = list(uber_files_dir.glob('**/*.csv'))

    li = [pd.read_csv(fnm, index_col=None, header=0) for fnm in tqdm(all_files)]

    frame = pd.concat(li, axis=0, ignore_index=True)

    frame_daily = frame.drop_duplicates()

    frame_daily = frame_daily.dropna(axis=0, how='all', subset=['Daily Mean Travel Time (Seconds)'])

    frame_daily.to_csv(result_fpath, index=False)


def gen_segments_length_file(shape_path: str, routes_path: str, routes_length_path: str) -> None:
    if Path(routes_length_path).is_file():
        print(f'File {routes_length_path} is already exists.')
        return

    road_segments = gpd.read_file(shape_path)
    routes = json.load(open(routes_path, 'r'))

    sid_length = pd.Series(road_segments.length_1.values, index=road_segments.num).to_dict()
    for routno, route in routes.items():
        for direction in ['to_center', 'from_center']:
            routes[routno][direction] = list(map(sid_length.get, routes[routno][direction]))

    with open(routes_length_path, 'w') as fp:
        json.dump(routes, fp)


def gen_neighbors_from_routes(routes_path: str, sid_neigh_path: str) -> None:
    if Path(sid_neigh_path).is_file():
        print(f'File {sid_neigh_path} is already exists.')
        return

    routes = json.load(open(routes_path, 'r'))

    sid_neighbors = {'to_center': {},
                     'from_center': {}
                     }

    for routno, route in routes.items():
        for direction in route:
            for idx, sid in enumerate(route[direction]):
                sid_neighbors[direction][sid] = (route[direction][idx - 1] if idx > 0 else None,
                                                 route[direction][idx + 1] if idx < len(
                                                     route[direction]) - 1 else None)

    with open(sid_neigh_path, 'w') as fp:
        json.dump(sid_neighbors, fp)
