import numpy as np
import pandas as pd
from utils import *
from data_preparation import *
from statsmodels.tsa.statespace.sarimax import SARIMAX

class STARIMA:
    def __init__(self, points, grid, last_train_date, last_test_date, temporal_granularity='1D'):
        self.point = points
        self.grid = grid
        self.end_date = last_train_date 
        self.last_test_day = last_test_date
        self.steps = 8*(pd.to_datetime(self.last_test_day) - pd.to_datetime(self.end_date)).days
        self.temporal_granularity = temporal_granularity

    def train(self):
        aggregated_data = aggregate_data(self.point, self.grid, self.temporal_granularity)
        aggregated_data = aggregated_data.reindex(columns=self.grid['cell'], fill_value=0)
        self.predictions = train_and_predict_sarima(aggregated_data, self.steps)

    def predict(self, idx_test):
        predictions_df = self.predictions.iloc[idx_test]/self.predictions.iloc[idx_test].sum()
        return predictions_df.to_numpy()