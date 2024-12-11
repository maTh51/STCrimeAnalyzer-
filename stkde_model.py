import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.neighbors import KernelDensity
from experimental.grid import set_points_to_grid, create_grid
from tqdm import tqdm

class STKDEModel:
    def __init__(
        self, points=None, grid=None, last_train_date=None, grid_size_optimization = None, municipalities = None, bandwidth=1.0, kernel="gaussian", days_past=28, temporal_granularity='1D', data_volume=1, static=False
    ):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.points = points.set_index('data_hora_fato')
        self.grid = grid
        self.days_past = days_past
        self.grid_size_optimization = grid_size_optimization
        self.municipalities = municipalities
        self.params = None
        self.grid2 = None
        self.data_volume = data_volume
        self.temporal_granularity = temporal_granularity[-1]
        self.period = int(temporal_granularity[:-1]) if self.temporal_granularity != 'D' else 24
        self.last_train_date = last_train_date
        self.static = static

    def fit(self, XY, weights=None):
        self.kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
        self.kde.fit(XY, sample_weight=weights)

    def predict(self, date: str):

        train_points = self.get_train(date)
        grid_idx = set_points_to_grid(train_points, self.grid).values

        XY = np.array(
            [self.grid.loc[grid_idx].centroid.x, self.grid.loc[grid_idx].centroid.y]
        ).T

        weights = self.get_weights(train_points, date)
        self.fit(XY, weights)

        XYgrid = np.array([self.grid.centroid.x, self.grid.centroid.y]).T

        grid_densities = np.exp(self.kde.score_samples(XYgrid))
        grid_probs = grid_densities / grid_densities.sum()

        return grid_probs

    def get_train(self, date: str):

        if self.static:
            start = (pd.to_datetime(self.last_train_date) - timedelta(days=self.days_past)).strftime(
                "%Y-%m-%d"
            )
            train_points = self.points.query(f"'{start}' <= data_hora_fato <= '{self.last_train_date}'").copy()
        else:
            start = (pd.to_datetime(date) - timedelta(days=self.days_past)).strftime(
                "%Y-%m-%d"
            )
            train_points = self.points.query(f"'{start}' <= data_hora_fato < '{date}'").copy()
        sz = int(train_points.shape[0]*self.data_volume)
        return train_points.sample(sz)

    def get_tdiff(self, points, date):
        date = pd.to_datetime(date)

        return (date - points.index).days * (24//self.period)\
                + ( \
                    (24//self.period)-1 \
                    - (points.index.hour // self.period) \
                    + date.hour // self.period \
                ) \
                + 1 \

    def get_prop(self, timegrid):

        prop = np.zeros((int(timegrid["tdiff"].max() + 1), int(self.grid2.shape[0])))
        tmp = (
            timegrid.groupby(["tdiff", "grid"]).size()
            / timegrid.groupby(["tdiff"]).size()
        ).reset_index()

        for _, row in tmp.iterrows():
            prop[int(row["tdiff"])][int(row["grid"])] = row[0]

        return prop

    def get_autocorr(self, prop):
        prop = prop.T
        autocorr = np.zeros(prop.shape)

        for s in range(prop.shape[0]):
            for shift in range(prop.shape[1]):
                if shift == 0:
                    autocorr[s][shift] = 1
                else:
                    autocorr[s][shift] = np.corrcoef(prop[s][:-shift], prop[s][shift:])[
                        0, 1
                    ]

        return autocorr

    def weight_function(self, t, p):
        p1, p2, p3, p4 = p
        T1, T2 = 24//self.period, 168//self.period
        return np.power(p1, t) + np.power(p2, t) * np.power(
            p3, np.sin(np.pi * t / T1) ** 2
        ) * np.power(p4, np.sin(np.pi * t / T2) ** 2)


    def gradient(self, X, Y, p):

        p1, p2, p3, p4 = p
        T1, T2 = 24//self.period, 168//self.period    
        errors = self.weight_function(X, p) - Y
        grad = np.zeros_like(p)

        grad[0] = np.sum(errors * X * p1 ** (X - 1))
        grad[1] = np.sum(
            errors
            * X
            * p2 ** (X - 1)
            * (p3 ** np.sin(np.pi * X / T1) ** 2)
            * p4 ** np.sin(np.pi * X / T2) ** 2
        )
        grad[2] = np.sum(
            errors
            * p2**X
            * (np.sin(np.pi * X / T1) ** 2)
            * (p3 ** ((np.sin(np.pi * X / T1) ** 2) - 1))
            * p4 ** np.sin(np.pi * X / T2) ** 2
        )
        grad[3] = np.sum(
            errors
            * p2**X
            * p3 ** np.sin(np.pi * X / T1) ** 2
            * (np.sin(np.pi * X / T2) ** 2)
            * (p4 ** ((np.sin(np.pi * X / T2) ** 2) - 1))
        )

        return grad / X.shape[0]

    def optimize_parameters(self, autocorr: np.ndarray, it: int, alpha: float):

        n_sector = self.grid2.shape[0]
        params = [0] * n_sector

        for s in tqdm(range(n_sector)):

            XY = np.array(
                [np.array([i, a]) for i, a in enumerate(autocorr[s]) if a >= 0]
            )
            X, Y = XY.T

            p = [0.95, 0.9995, 0.001, 0.145]  # Parametros base do artigo

            for _ in range(it):
                grad = self.gradient(X, Y, p)
                if np.isnan(
                    np.sum((self.weight_function(X, (p - alpha * grad)) - Y) ** 2)
                    / X.shape[0]
                ):
                    break
                p -= alpha * grad
            params[s] = p

        return params

    def get_weights(self, points, date):

        tdiff = self.get_tdiff(points, date)
        if self.grid2 is None:
            self.grid2 = create_grid(
                self.grid_size_optimization, self.municipalities
            )  # Mudar para o nome do municipio
            pgrid = set_points_to_grid(points, self.grid2)

            tmp = np.array([tdiff, pgrid]).T
            timegrid = pd.DataFrame(tmp, columns=["tdiff", "grid"])
            # if self.params is None:
            prop = self.get_prop(timegrid)
            autocorr = self.get_autocorr(prop)
            self.params = self.optimize_parameters(autocorr, 1000, 0.01)
        else:
            pgrid = set_points_to_grid(points, self.grid2)
            tmp = np.array([tdiff, pgrid]).T
            timegrid = pd.DataFrame(tmp, columns=["tdiff", "grid"])
            

        weights = [
            self.weight_function(row["tdiff"], self.params[row["grid"]])
            for _, row in timegrid.iterrows()
        ]
        weights = np.array(weights).clip(0, 1)

        weights[np.isnan(weights)] = 0
        return weights
    
    def train(self):
        pass
