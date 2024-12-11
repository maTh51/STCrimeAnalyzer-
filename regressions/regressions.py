import numpy as np
import pandas as pd
import datetime
from data_handler import *

class REGRESSIONS():
    def __init__(self, points, grid, last_train_date, last_test_date, model_name, n_estimators, max_depth, temporal_granularity='1d') -> None:
        self.points=points
        self.grid=grid
        self.model_name=model_name
        self.last_train_date = last_train_date 
        self.last_test_date = last_test_date
        self.temporal_granularity = temporal_granularity
        # self.period = int(temporal_granularity[:-1]) if self.temporal_granularity[-1].lower() != 'd' else 24
        # self.steps = 8*(pd.to_datetime(self.last_test_day) - pd.to_datetime(self.end_date)).days
        # self.period = 24//self.period

        self.model = create_model(model_name, n_estimators, max_depth)
        self.features = [
            'night_or_day', 'cos_hour', 'cos_day_of_week', 'day_of_week', 'sin_minute',
            'sin_day_of_week', 'day', 'month', 'hour', 'cos_minute', 
            'cos_month', 'year', 'sin_month', 'sin_hour'
        ]
        self.target = ['numero_latitude', 'numero_longitude']  

    def train(self):
        X_train, y_train = create_features(self.points, self.features, self.target)
        self.model.fit(X_train, y_train)       

    def predict(self, test_points):
        test_points = test_points.reset_index()
        X_test, y_test = create_features(test_points, self.features, self.target)
        predictions = self.model.predict(X_test)

        return convert_to_matrix(predictions, self.grid)
