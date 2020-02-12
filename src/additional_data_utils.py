import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


def clean_train(train_path: str,
                train_cleaned_path: str,
                inj_path: str,
                veh_path: str
                ) -> None:
    if Path(train_cleaned_path).is_file():
        print(f'File {train_cleaned_path} is already exists.')
        return

    train_raw = pd.read_csv(train_path, parse_dates=['Occurrence Local Date Time'])
    orig_cols = train_raw.columns.values

    inj = pd.read_csv(inj_path, parse_dates=['Created Local Date Time'])
    inj.drop_duplicates('Event Id', inplace=True)

    veh = pd.read_csv(veh_path, parse_dates=['CreatedLOcalDateTime'])
    veh.drop_duplicates('EventID', inplace=True)

    train_cleansed = train_raw.merge(veh, left_on='EventId', right_on='EventID', how='left')
    train_cleansed = train_cleansed.merge(inj, left_on='EventId', right_on='Event Id', how='left')

    train_cleansed['CreatedLOcalDateTime'] = train_cleansed['CreatedLOcalDateTime'].combine_first(
        train_cleansed['Created Local Date Time'])

    index = ((train_cleansed['Occurrence Local Date Time'].dt.time == datetime.time(hour=0, minute=0, second=0))
             & (train_cleansed['Occurrence Local Date Time'].dt.date == train_cleansed[
                'CreatedLOcalDateTime'].dt.date))

    train_cleansed['Occurrence Local Date Time'][index] = train_cleansed['CreatedLOcalDateTime']

    train_cleansed[orig_cols].to_csv(train_cleaned_path, index=False)

    print(f'{train_cleaned_path} was created.')


def prepare_weather(weather_68816_path: str,
                    weather_fact_path: str,
                    result_weather_path: str
                    ) -> None:
    if Path(result_weather_path).is_file():
        print(f'File {result_weather_path} is already exists.')
        return

    weather = pd.read_csv(weather_fact_path,
                          sep=";", skiprows=6, usecols=range(13),
                          parse_dates=['Local time in Cape Town (airport)'])

    weather_add = pd.read_csv(weather_68816_path,
                              sep=";", skiprows=6, index_col=False, usecols=range(25),
                              parse_dates=['Local time in Cape Town (airport)'])

    weather = weather[['Local time in Cape Town (airport)', 'T', 'P0', 'P', 'U', 'DD', 'Ff', 'VV', 'Td']]
    weather_add = weather_add[['Local time in Cape Town (airport)', 'ff10', 'N', 'WW', 'Cl',
                               'Nh', 'H', 'Cm', 'Ch', 'RRR', 'tR']]

    weather['Local time in Cape Town (airport)'] = weather['Local time in Cape Town (airport)'].dt.round('H')
    weather = weather.drop_duplicates('Local time in Cape Town (airport)')

    weather['DD'] = weather['DD'].apply(lambda s: str(s).replace('Wind blowing from the ', ''))
    weather['DD'] = weather['DD'].replace('nan', np.nan)

    weather['Td'] = weather['Td'].replace('>Vertical visibility less than 984 feet</span>', np.nan).astype('float')

    weather['VV'] = weather['VV'].replace(
        {'>Vertical visibility less than 30 m</span><span class=h_1" style="display: none': '0.1',
         'Broken clouds': '10.0',
         '10.0 and more': '10.0',
         r"[a-zA-Z</>=_:()%-]": '',
         "\"": '',
         ' 6090 000 3270  0000 000 750  1': '10.0'}, regex=True)
    weather['VV'] = weather['VV'].astype(float)

    dts = pd.date_range('2016-01-01',
                        '2019-03-31 23:00:00',
                        freq="1h")
    w = pd.DataFrame({'datetime': dts})
    w = w.merge(weather, left_on='datetime', right_on='Local time in Cape Town (airport)', how='left')
    w = w.merge(weather_add, left_on='datetime', right_on='Local time in Cape Town (airport)', how='left')

    w.drop(['Local time in Cape Town (airport)_x', 'Local time in Cape Town (airport)_y'], inplace=True, axis=1)

    w = w.fillna(method='bfill', axis=0, limit=1)
    w = w.fillna(method='ffill', axis=0, limit=1)

    w.columns = ['datetime', 'temp', 'P0', 'P', 'humidity', 'wind_dir', 'wind_speed', 'visibility', 'dewpoint',
                 'max_gust',
                 'cloud_cover', 'weather_cond', 'cloud_1', 'cloud1_cover', 'cloud_height', 'cloud_2', 'cloud_3',
                 'precip_mm', 'precip_time']

    w['cloud_cover_fog'] = (w.cloud_cover == 'Sky obscured by fog and/or other meteorological phenomena.').astype(int)

    cloud_cover_dict = {'no clouds': 0,
                        '10%  or less, but not 0': 0.05,
                        '20–30%.': 0.25,
                        '90  or more, but not 100%': 0.95,
                        '70 – 80%.': 0.75,
                        '50%.': 0.5,
                        '60%.': 0.6,
                        '40%.': 0.4,
                        '100%.': 1,
                        'Sky obscured by fog and/or other meteorological phenomena.': np.nan
                        }
    w['cloud_cover'] = w.cloud_cover.replace(cloud_cover_dict).astype(float)
    w['cloud1_cover'] = w.cloud1_cover.replace(cloud_cover_dict).astype(float)

    w['wind_dir_angle'] = w.wind_dir.replace({'south': 180,
                                              'south-southeast': 157.5,
                                              'south-southwest': 202.5,
                                              'north-west': 315,
                                              'variable wind direction': np.nan,
                                              'north': 0,
                                              'south-west': 225,
                                              'north-northwest': 337.5,
                                              'north-northeast': 22.5,
                                              'south-east': 135,
                                              'west-northwest': 292.5,
                                              'west-southwest': 247.5,
                                              'north-east': 45,
                                              'west': 270,
                                              'east-southeast': 112.5,
                                              'east-northeast': 67.5,
                                              'east': 90,
                                              'Calm, no wind': np.nan}).astype(float)

    w['wind_dir_defined'] = (~w.wind_dir.isin(['variable wind direction', 'Calm, no wind']) &
                             ~w.wind_dir.isnull()).astype(int)

    w['precip_mm'] = w.precip_mm.replace({'No precipitation': 0,
                                          'Trace of precipitation': 0.25}).astype(float)

    w['cloud_height'] = w.cloud_height.replace({'600-1000': 800,
                                                '1000-1500': 1250,
                                                '2500 or more, or no clouds.': 2500,
                                                '300-600': 450,
                                                '1500-2000': 1750,
                                                '100-200': 150,
                                                '200-300': 250,
                                                '50-100': 75,
                                                '2000-2500': 2250,
                                                'Less than  50': 50}).astype(float)

    w['month'] = w.datetime.dt.month
    w['quarter'] = w.datetime.dt.year + ((w.month - 1) // 3) * 0.25
    w['week'] = w.datetime.dt.week
    w['hour'] = w.datetime.dt.hour

    fill_nan_mean = ['temp', 'P0', 'P', 'humidity', 'wind_speed', 'visibility',
                     'max_gust', 'dewpoint', 'cloud_cover', 'wind_dir_angle',
                     'cloud_height', 'cloud1_cover']
    fill_nan_most_common = ['wind_dir']
    fill_nan_unknown = ['cloud_1', 'cloud_2', 'cloud_3', 'weather_cond']
    fill_nan_0 = ['cloud1_cover', 'cloud_cover', 'precip_mm', 'precip_time', 'wind_dir_angle']
    fill_nan_max = ['cloud_height']

    for f in fill_nan_mean:
        w[f] = w[f].fillna(w.groupby(['quarter', 'month', 'hour'])[f].transform('mean'))

    for f in fill_nan_most_common:
        w[f] = w[f].fillna(w.groupby(['quarter', 'month', 'hour'])[f].transform(lambda x: x.value_counts().idxmax()))

    for f in fill_nan_unknown:
        w[f] = w[f].fillna('unknown')

    for f in fill_nan_0:
        w[f] = w[f].fillna(0)

    for f in fill_nan_max:
        w[f] = w[f].fillna(w[f].max())

    w.weather_cond = w.weather_cond.apply(lambda s: s.lower())
    conditions = ['mist', 'fog', 'smoke', 'rain', 'drizzle', 'snow']
    for condition in conditions:
        w[condition] = w.weather_cond.str.contains(condition).astype(int)

    w.drop(['month', 'quarter', 'week', 'hour'], axis=1).to_csv(result_weather_path, index=False)


def prepare_sanral(sanral_v3_path: Path,
                   result_hourly_path: Path
                   ) -> None:
    if Path(result_hourly_path).is_file():
        print(f'File {result_hourly_path} is already exists.')
        return

    exceptions = [sanral_v3_path / p for p in [
        '2017/WC May 2017 Hourly Lane Spec.csv',
        '2017/WC June 2017 Hourly Lane spec.csv',
        '2017/WC March 2017 Hourly.csv',
        '2018/12. WC December 2018 Hourly.csv',
        '2018/11. WC November 2018 Hourly.csv']]

    li = []
    for year in ['2016', '2017', '2018', '2019']:

        all_files = list((sanral_v3_path / year).glob('**/*.csv'))

        for filename in all_files:
            if filename in exceptions:
                continue
            df = pd.read_csv(filename,
                             sep=None, engine='python',
                             names=['Region', 'Site name', 'Date of Collection Period', 'Hour of Collection Period',
                                    'Vehicle Class Type', 'Total Count of Vehicle Class', 'Average Speed'])
            li.append(df)

    vds_hourly = pd.concat(li, axis=0, ignore_index=True)

    vds_hourly = pd.pivot_table(vds_hourly, values=['Total Count of Vehicle Class', 'Average Speed'],
                                index=['Site name', 'Date of Collection Period', 'Hour of Collection Period'],
                                columns='Vehicle Class Type', aggfunc=np.sum).reset_index().droplevel(1, axis=1)

    vds_hourly.columns = ['Site name', 'Date of Collection Period',
                          'Hour of Collection Period', 'Average Speed', 'Average Speed', 'Average Speed',
                          'Count of Vehicle Class 1', 'Count of Vehicle Class 2', 'Count of Vehicle Class 3']

    vds_hourly = vds_hourly.loc[:, ~vds_hourly.columns.duplicated()]  # type: ignore

    vds_hourly.sort_values(['Date of Collection Period', 'Hour of Collection Period']).to_csv(result_hourly_path,
                                                                                              index=False)
