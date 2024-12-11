import pandas as pd
import geopandas as gpd
from experimental.grid import points_in_municipalities


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