import numpy as np
import pandas as pd
from datetime import timedelta
from experimental.grid import set_points_to_grid, matrix_to_grid


class EvaluationModel:

    def __init__(self, models, points, grid, start_date, end_date, days_test=1, temporal_granularity='1D', grid_size = 1000):
        self.models = models
        self.points = points.set_index('data_hora_fato')
        self.grid = grid
        self.start_date = pd.to_datetime(start_date) + timedelta(days=1)
        self.end_date = pd.to_datetime(end_date)
        self.days_test = days_test
        self.temporal_granularity = temporal_granularity
        self.grid_size = grid_size

    def simulate(self, hit_rate_percentage: float):

        self.hit_rate_percentage = hit_rate_percentage
        date = self.start_date
        results = []
        for idx_test, dt in enumerate(list(self.points.resample(self.temporal_granularity))):
            day, data = dt
            for model in self.models:
                if model.__class__.__name__ == "STARIMA":
                    predicted_grid = model.predict(idx_test)
                elif model.__class__.__name__ == "REGRESSIONS":
                    predicted_grid = model.predict(data)
                else:    
                    predicted_grid = model.predict(date)

                predicted_grid = matrix_to_grid(predicted_grid, self.grid)
                test_grid = self.get_test_grid(data, self.grid)
                eval = self.eval(predicted_grid, test_grid)
                eval.update({"model": model.__class__.__name__})
                results.append(eval)

            date += timedelta(days=self.days_test)
            if(date > self.end_date): break


        return pd.DataFrame(results)

    def eval(self, grid_predicted, grid_test):

        grid_predicted = grid_predicted.sort_values("cell")
        grid_test = grid_test.sort_values("cell")

        metrics = {
            f"HR({self.hit_rate_percentage*100}%)": self.hit_rate(grid_predicted, grid_test),
            "ALS": self.average_logarithmic_score(grid_predicted, grid_test)[0],
            "ALS_zeros": self.average_logarithmic_score(grid_predicted, grid_test)[1],
            "MSE": self.mean_squared_error(grid_predicted, grid_test),
            "PAI": self.pai(grid_predicted, grid_test),
            "PEI": self.pei(grid_predicted, grid_test)
        }

        return metrics

    def get_test_grid(self, points, grid):

        points["cell"] = set_points_to_grid(points=points, grid=grid).values
        test_grid = grid.copy()
        if "count" in test_grid.columns:
            test_grid.drop(columns=["count"], inplace=True)
        test_grid = grid.merge(
            points.groupby("cell").size().reset_index(name="count"),
            on="cell",
            how="left",
        ).fillna(0)

        test_grid['count'] = test_grid['count'] / test_grid['count'].sum()

        return test_grid

    def hit_rate(self, grid_predicted, grid_test, ncells=0):
        if not ncells:
            num_cells = int(len(grid_predicted) * (self.hit_rate_percentage))
            top_cells = grid_predicted.nlargest(num_cells, "count")
        else:
            top_cells = grid_predicted.nlargest(ncells, "count")

        hit_rate = (
            grid_test[grid_test["cell"].isin(top_cells["cell"])]["count"].sum()
            / grid_test["count"].sum()
        )

        return hit_rate

    def average_logarithmic_score(self, grid_predicted, grid_test):
        val = grid_predicted["count"].to_numpy()[
            grid_test[grid_test["count"] > 0]["cell"].values
        ]
        zeros = (val == 0).sum()
        val[val == 0] = 1
        return np.mean(np.log(val)), zeros

    def mean_squared_error(self, grid_predicted, grid_test):
        return np.mean((grid_predicted["count"] - grid_test["count"]) ** 2)

    def pai(self, grid_predicted, grid_test, ncells=0):

        hr = self.hit_rate(grid_predicted, grid_test, ncells)

        num_cells = int(len(grid_predicted) * (self.hit_rate_percentage))
        grid_area = self.grid_size * self.grid_size

        a = num_cells * grid_area
        A = len(grid_predicted) * grid_area

        return (hr / (a/A))

    def pei(self, grid_predicted, grid_test):

        
        num_cells = grid_test[grid_test["count"] > 0]["cell"].nunique()
        max_cells = int(len(grid_predicted) * (self.hit_rate_percentage))

        num_cells = min(max_cells, num_cells)

        pai_ = self.pai(grid_predicted, grid_test, ncells=num_cells)

        grid_area = self.grid_size * self.grid_size
        a = num_cells * grid_area
        A = len(grid_predicted) * grid_area


        return pai_ / (1/(a/A))
    