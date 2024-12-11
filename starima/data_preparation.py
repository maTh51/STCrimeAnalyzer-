import pandas as pd
import geopandas as gpd
from typing import Tuple

def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    cols_wanted = ['data_hora_fato', 'natureza_descricao', 'numero_latitude', 'numero_longitude']
    df = pd.read_csv(file_path, usecols=cols_wanted)
    df['data_hora_fato'] = pd.to_datetime(df['data_hora_fato'], errors='coerce')
    df.dropna(inplace=True)
    df = df[df['data_hora_fato'].dt.year == 2024]
    df = df[df['natureza_descricao'] == 'FURTO']
    df.sort_values(by='data_hora_fato', inplace=True)
    start_date = df['data_hora_fato'].min()
    end_date = start_date + pd.DateOffset(months=7)

    return df[(df['data_hora_fato'] >= start_date) & (df['data_hora_fato'] < end_date)]

def aggregate_data(data: pd.DataFrame, grid: gpd.GeoDataFrame, temporal_granularity) -> pd.DataFrame:
    data['data_hora_fato'] = pd.to_datetime(data['data_hora_fato'], errors='coerce')
    data['geometry'] = gpd.points_from_xy(data['numero_longitude'], data['numero_latitude'])
    data_gdf = gpd.GeoDataFrame(data, geometry='geometry', crs='EPSG:4326')
    
    if grid.crs.to_string() != data_gdf.crs.to_string():
        grid = grid.to_crs(data_gdf.crs)

    joined = gpd.sjoin(data_gdf, grid, how='inner', predicate='intersects')

    if joined.empty:
        return pd.DataFrame()
    
    joined['data_hora_fato'] = pd.to_datetime(joined['data_hora_fato'])
    joined.set_index('data_hora_fato', inplace=True)
    grouped = joined.groupby(['cell', pd.Grouper(freq=temporal_granularity)]).size()

    return grouped.unstack(level=0, fill_value=0)
