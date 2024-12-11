import jaydebeapi
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
from tqdm import tqdm
from datetime import timedelta
import pandas as pd
import geopandas as gpd

from grid import transform_crs, select_municipalities

def read_df_offline(config, df_path):
    df_path = './data/concatenado1.csv'
    df = pd.read_csv(df_path)
    df = df[config['database']['columns']]
    df['data_hora_fato'] = pd.to_datetime(df['data_hora_fato'])
    df = df.query(f"'{config['database']['filters']['start_date']}'\
                    <= data_hora_fato <= \
                    '{config['database']['filters']['end_date']}'")
    return df

def pre_process(
    data: pd.DataFrame, neighborhood: list, columns: list
) -> gpd.GeoDataFrame:

    points = data.dropna()
    points = points_in_municipalities(neighborhood, data)
    points["data_hora_fato"] = pd.to_datetime(points["data_hora_fato"])
    points = points[columns + ["geometry"]]

    return points

def train_test_split(points: pd.DataFrame, train_end_date: str, test_end_date) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = points.query(f"data_hora_fato <= '{train_end_date}'").copy()
    test  = points.query(f"'{train_end_date}' < data_hora_fato <= '{test_end_date}'").copy()
    
    return train, test


def points_in_municipalities(municipalities, data):
    points = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data["numero_longitude"], data["numero_latitude"]),
    )
    points = transform_crs(points, 4326, 3857)

    grid = select_municipalities(municipalities)

    return gpd.sjoin(points, grid, predicate="within")