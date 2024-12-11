from datetime import datetime
import pandas as pd
import numpy as np
import geopandas as gpd

def string2binary(strings, T):
    binary_timestamps = []
    for string in strings:
        ts = datetime.strptime(string, '%Y-%m-%d %H:%M:%S')
        year, month, day = ts.year, ts.month, ts.day
        slot = int((ts.hour * 60 + ts.minute) / (60 * (24 / T))) + 1
        binary_timestamp = f'{year:04d}{month:02d}{day:02d}{slot:02d}'.encode('utf-8')
        binary_timestamps.append(binary_timestamp)
    return binary_timestamps

def remove_outliers(df):
    limite_inferior = df['numero_latitude'].quantile(0.003)
    limite_superior = df['numero_latitude'].quantile(0.997)
    df = df[(df['numero_latitude'] >= limite_inferior) & (df['numero_latitude'] <= limite_superior)]
    limite_inferior = df['numero_longitude'].quantile(0.003)
    limite_superior = df['numero_longitude'].quantile(0.997)
    df = df[(df['numero_longitude'] >= limite_inferior) & (df['numero_longitude'] <= limite_superior)]
    return df

def create_bins(df,map_height, map_width):
    bins = pd.cut(df['numero_latitude'], bins=map_height, include_lowest=True, labels=False)
    df['Latitude'] = bins
    bins_lon = pd.cut(df['numero_longitude'], bins=map_width, include_lowest=True, labels=False)
    df['Longitude'] = bins_lon
    return df

def section_time(df,T,temporal_granularity='1D'):
    # df["data_hora_fato"] = pd.to_datetime(df["data_hora_fato"])
    # if T == 1:
    #     time = '1d'
    # elif T == 3:
    #     time = '8h'
    df['data_hora_fato'] = df['data_hora_fato'].dt.floor(f'{temporal_granularity}')
    df['data_hora_fato'] = df['data_hora_fato'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return df

def get_arrays_and_timestamps(df,map_height, map_width, T):
    grupos = df.groupby('data_hora_fato')
    group_identifiers = grupos.groups
    lista_dataframes = [grupo.copy() for _, grupo in grupos]
    timestamps=string2binary(list(group_identifiers.keys()),T)
    timestamps=np.array(timestamps)
    crimes_by_timestamp=[]
    array_shape = (map_height,map_width)
    
    for temp_df in lista_dataframes:
        crime_array, _, _ = np.histogram2d(
            temp_df['Latitude'], 
            temp_df['Longitude'], 
            bins=[array_shape[0], array_shape[1]], 
            range=[[0, map_height], [0, map_width]],
            density=False
        )
        crime_array = crime_array.astype(int)
        crimes_by_timestamp.append([crime_array, crime_array])
    data=np.array(crimes_by_timestamp)
    return data, timestamps

def load_crime(df,map_height, map_width,T,temporal_granularity='1D'):
    # df= remove_outliers(df)
    df= create_bins(df, map_height, map_width)
    df= section_time(df,T,temporal_granularity)
    data, timestamps = get_arrays_and_timestamps(df, map_height, map_width, T)
    return data, timestamps

