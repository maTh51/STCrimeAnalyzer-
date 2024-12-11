import numpy as np
import pandas as pd

from experimental.st3dnet.st3dnet import train_st3dnet

class ST3DNETModel():
    def __init__(self, points, grid, last_train_date, len_closeness, len_period, len_trend, temporal_granularity='1d') -> None:
        self.point=points
        self.grid=grid
        self.len_closeness=len_closeness
        self.len_period=len_period
        self.len_trend=len_trend
        self.end_date=last_train_date
        self.temporal_granularity = temporal_granularity
        self.period = int(temporal_granularity[:-1]) if self.temporal_granularity[-1].lower() != 'd' else 24
        self.period = 24//self.period


    def train(self):
        _, _, self.model = train_st3dnet(self.point, self.grid,
                                   self.len_closeness,self.len_period,self.len_trend,self.period,self.temporal_granularity)
        

    def predict(self, date):
        date = pd.to_datetime(date)
        len_test = (date - pd.to_datetime(self.end_date)).days*self.period + date.hour//(self.period*24)
        X_test_c = np.full((len_test, 2, self.len_closeness, self.grid["ycell"].max() + 1, self.grid["xcell"].max() + 1), -1)
        X_test_t = np.full((len_test, 2, self.len_trend, self.grid["ycell"].max() + 1, self.grid["xcell"].max() + 1), -1)
        X_test = [X_test_c, X_test_t]
        result = self.model.predict(X_test)
        predicts = []
        for i in range(len(result)):
            sum_predict = result[i][0] + result[i][1]
            predicts.append(sum_predict)

        predicts = np.array(predicts)

        rotated_predicts = np.flipud(np.fliplr(predicts))
        return rotated_predicts[-1]