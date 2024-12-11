import numpy as np
import geopandas as gpd


class StaticCountModel:
    def __init__(self, matrix, points: gpd.GeoDataFrame = None) -> None:
        self.points = points
        self.matrix = matrix

    def train(self):

        self.train_matrix = np.sum(self.matrix, axis=0)

        return self.train_matrix

    def predict(self, date):
        return self.train_matrix
