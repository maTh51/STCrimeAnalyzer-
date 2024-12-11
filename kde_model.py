import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.neighbors import KernelDensity
from experimental.grid import set_points_to_grid


class KDEModel:
    def __init__(
        self, points=None, grid=None, bandwidth=1.0, kernel="gaussian", days_past=28
    ):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.points = points
        self.grid = grid
        self.days_past = days_past

    def fit(self, XY):
        self.kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
        self.kde.fit(XY)

    def predict(self, date: str):

        train_points = self.get_train(date)
        grid_idx = set_points_to_grid(train_points, self.grid).values

        XY = np.array(
            [self.grid.loc[grid_idx].centroid.x, self.grid.loc[grid_idx].centroid.y]
        ).T
        self.fit(XY)

        XYgrid = np.array([self.grid.centroid.x, self.grid.centroid.y]).T

        grid_densities = np.exp(self.kde.score_samples(XYgrid))
        grid_probs = grid_densities / grid_densities.sum()

        return grid_probs

    def get_train(self, date: str):
        start = (pd.to_datetime(date) - timedelta(days=self.days_past)).strftime(
            "%Y-%m-%d"
        )
        train_points = self.points.query(f"'{start}' <= data_hora_fato < '{date}'")
        return train_points
