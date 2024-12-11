import jaydebeapi
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, Point
from tqdm import tqdm
from datetime import timedelta
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

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

def create_model(self, model_name, n_estimators=2, max_depth=33):
    match model_name:
        case "Random Forest Regressor":
            return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        case "Extra Trees Regressor":
            return ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        case "Decision Tree Regressor":
            return DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        case "Bagging Regressor":
            return BaggingRegressor(n_estimators=n_estimators, random_state=42)
        case _:
            return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

def create_features(df, features, target):
    df['year'] = df['data_hora_fato'].dt.year
    df['month'] = df['data_hora_fato'].dt.month
    df['day'] = df['data_hora_fato'].dt.day
    df['day_of_week'] = df['data_hora_fato'].dt.dayofweek
    df['hour'] = df['data_hora_fato'].dt.hour
    df['minute'] = df['data_hora_fato'].dt.minute
    df['night_or_day'] = df['hour'].apply(lambda x: 1 if x < 6 or x >= 18 else 0)


    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['sin_minute'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['cos_minute'] = np.cos(2 * np.pi * df['minute'] / 60)
    df['sin_day_of_week'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['cos_day_of_week'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)

    X = df[features]
    y = df[target]

    return X, y

def convert_to_matrix(predictions, grid):
    df_predicoes = pd.DataFrame(predictions, columns=["numero_latitude", "numero_longitude"])

    gdf_predicoes = gpd.GeoDataFrame(
        df_predicoes,
        geometry=df_predicoes.apply(lambda row: Point(row["numero_longitude"], row["numero_latitude"]), axis=1),
        crs="EPSG:4326"
    )

    gdf_predicoes = gdf_predicoes.to_crs(grid.crs)
    gdf_predicoes_com_grid = gpd.sjoin(gdf_predicoes, grid, how='left', predicate='within')

    df_counts = gdf_predicoes_com_grid.groupby(['xcell', 'ycell']).size().reset_index(name='count')

    max_xcell = grid['xcell'].max()
    max_ycell = grid['ycell'].max()
    full_index_x = np.arange(0, max_xcell + 1)
    full_index_y = np.arange(0, max_ycell + 1)
    
    matriz_contagem = pd.pivot_table(
        df_counts, 
        values='count', 
        index='ycell', 
        columns='xcell', 
        fill_value=0
    ).reindex(index=full_index_y, columns=full_index_x, fill_value=0).to_numpy()

    return matriz_contagem
