from datetime import datetime
import pandas as pd
import numpy as np
import geopandas as gpd
from geopandas.tools import sjoin
import time
import tensorflow as tf

from experimental.st3dnet.ST3DNet import *
from experimental.st3dnet.utils import *
from experimental.st3dnet.crime_data import load_crime
from experimental.st3dnet.data_train import main

def remove_incomplete_days(data, timestamps, T):
    days = [] 
    days_incomplete = []
    i = 0
    while i < len(timestamps):
        if int(timestamps[i][8:]) != 1:
            i += 1
        elif i+T-1 < len(timestamps) and int(timestamps[i+T-1][8:]) == T:
            days.append(timestamps[i][:8])
            i += T
        else:
            days_incomplete.append(timestamps[i][:8])
            i += 1
    days = set(days)
    idx = []
    for i, t in enumerate(timestamps):
        if t[:8] in days:
            idx.append(i)

    data = data[idx]
    timestamps = [timestamps[i] for i in idx]
    return data, timestamps

def timestamp2vec(timestamps):
    vec = [time.strptime(str(t[:8], encoding='utf-8'), '%Y%m%d').tm_wday for t in timestamps]
    ret = []
    for i in vec:
        v = [0 for _ in range(7)]
        v[i] = 1
        if i >= 5:
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday
        ret.append(v)
    return np.asarray(ret)

def string2timestamp(strings, T):
    timestamps = []
    time_per_slot = 24.0 / T
    for t in strings:
        year, month, day, slot = int(t[:4]), int(t[4:6]), int(t[6:8]), int(t[8:])-1
        timestamps.append(pd.Timestamp(datetime(year, month, day, hour=int(slot * time_per_slot), minute=0)))
    return timestamps

class STMatrix(object):
    def __init__(self, data, timestamps, T, CheckComplete=True):
        super(STMatrix, self).__init__()
        assert len(data) == len(timestamps)
        self.data = data
        self.data_1 = data[:, 0, :, :]
        self.data_2 = data[:, 1, :, :]
        self.timestamps = timestamps
        self.T = T
        self.pd_timestamps = string2timestamp(timestamps, T=self.T)
        if CheckComplete:
            self.check_complete()
        self.make_index()

    def make_index(self):
        self.get_index = dict()
        for i, ts in enumerate(self.pd_timestamps):
            self.get_index[ts] = i

    def check_complete(self):
        missing_timestamps = []
        offset = pd.DateOffset(minutes=24 * 60 // self.T)
        pd_timestamps = self.pd_timestamps
        i = 1
        while i < len(pd_timestamps):
            if pd_timestamps[i - 1] + offset != pd_timestamps[i]:
                missing_timestamps.append("(%s -- %s)" % (pd_timestamps[i - 1], pd_timestamps[i]))
            i += 1
        for v in missing_timestamps:
            print(v)
        assert len(missing_timestamps) == 0

    def get_matrix(self, timestamp):
        return self.data[self.get_index[timestamp]]

    def get_matrix_1(self, timestamp):  # in_flow
        ori_matrix = self.data_1[self.get_index[timestamp]]
        new_matrix = ori_matrix[np.newaxis, :]
        return new_matrix

    def get_matrix_2(self, timestamp):  # out_flow
        ori_matrix = self.data_2[self.get_index[timestamp]]
        new_matrix = ori_matrix[np.newaxis, :]
        return new_matrix
    
    def check_it(self, depends):
        for d in depends:
            if d not in self.get_index.keys():
                return False
        return True

    def create_dataset_3D(self, len_closeness, len_trend, TrendInterval=7, len_period=3, PeriodInterval=1):
        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
        XC = [] # closeness
        XP = [] # period
        XT = [] # trend
        Y = []
        timestamps_Y = []
        depends = [range(1, len_closeness + 1),
                   [PeriodInterval * self.T * j for j in range(1, len_period + 1)],
                   [TrendInterval * self.T * j for j in range(1, len_trend + 1)]]
        print("depends: ", depends)
    
        i = max(self.T * TrendInterval * len_trend, self.T * PeriodInterval * len_period, len_closeness)

        while i < len(self.pd_timestamps):
            Flag = True
            for depend in depends:
                if Flag is False:
                    break
                Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame for j in depend])
            if Flag is False:
                i += 1
                continue
            
            #closeness
            c_1_depends = list(depends[0])  # in_flow
            c_1_depends.sort(reverse=True)
            c_2_depends = list(depends[0])  # out_flow
            c_2_depends.sort(reverse=True)
            x_c_1 = [self.get_matrix_1(self.pd_timestamps[i] - j * offset_frame) for j in c_1_depends]
            x_c_2 = [self.get_matrix_2(self.pd_timestamps[i] - j * offset_frame) for j in c_2_depends]  

            x_c_1_all = np.vstack(x_c_1) 
            x_c_2_all = np.vstack(x_c_2)  
            x_c_1_new = x_c_1_all[np.newaxis, :] 
            x_c_2_new = x_c_2_all[np.newaxis, :] 

            x_c = np.vstack([x_c_1_new,x_c_2_new])
            p_depends = list(depends[1])

            # period
            if (len(p_depends) > 0):
                p_depends.sort(reverse=True)
                x_p_1 = [self.get_matrix_1(self.pd_timestamps[i] - j * offset_frame) for j in p_depends]
                x_p_2 = [self.get_matrix_2(self.pd_timestamps[i] - j * offset_frame) for j in p_depends]

                x_p_1_all = np.vstack(x_p_1)
                x_p_2_all = np.vstack(x_p_2)

                x_p_1_new = x_p_1_all[np.newaxis, :]
                x_p_2_new = x_p_2_all[np.newaxis, :]

                x_p = np.vstack([x_p_1_new,x_p_2_new]) 

            # trend
            t_depends = list(depends[2])
            if (len(t_depends) > 0):
                t_depends.sort(reverse=True)

                x_t_1 = [self.get_matrix_1(self.pd_timestamps[i] - j * offset_frame) for j in t_depends]
                x_t_2 = [self.get_matrix_2(self.pd_timestamps[i] - j * offset_frame) for j in t_depends]

                x_t_1_all = np.vstack(x_t_1) 
                x_t_2_all = np.vstack(x_t_2) 

                x_t_1_new = x_t_1_all[np.newaxis, :]
                x_t_2_new = x_t_2_all[np.newaxis, :]

                x_t = np.vstack([x_t_1_new,x_t_2_new]) 

            y = self.get_matrix(self.pd_timestamps[i])

            if len_closeness > 0:
                XC.append(x_c)
            if len_period > 0:
                XP.append(x_p)
            if len_trend > 0:
                XT.append(x_t)
            Y.append(y)
            timestamps_Y.append(self.timestamps[i])
            i += 1

        XC = np.asarray(XC)
        XP = np.asarray(XP)
        XT = np.asarray(XT)
        Y = np.asarray(Y)
        return XC, XP, XT, Y, timestamps_Y
    
def timestamp2vec(timestamps):
    vec = [time.strptime(str(t[:8], encoding='utf-8'), '%Y%m%d').tm_wday for t in timestamps]
    ret = []
    for i in vec:
        v = [0 for _ in range(7)]
        v[i] = 1
        if i >= 5:
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday
        ret.append(v)
    return np.asarray(ret)

def load_data(df,map_height, map_width, T, nb_flow=2, len_closeness=None,
               len_period=None, len_trend=None, len_test=None, meta_data=True, temporal_granularity='1D'):
    assert (len_closeness + len_period + len_trend > 0)
    data, timestamps = load_crime(df=df,map_height=map_height, map_width=map_width,T=T,temporal_granularity=temporal_granularity)
    data = data[:, :nb_flow]
    data[data < 0] = 0.
    data_all = [data]
    timestamps_all = [timestamps]
    data_train = data[:-len_test]
    print("len_test: ", len_test)
    print('data shape: ', data.shape)
    print('train_data shape: ', data_train.shape)

    mmn = MinMaxNormalization()
    mmn.fit(data_train)
    data_all_mmn = []
    for d in data_all:
        data_all_mmn.append(mmn.transform(d))
    XC, XP, XT = [], [], []
    Y = []
    timestamps_Y = []
    for data, timestamps in zip(data_all_mmn, timestamps_all):
        st = STMatrix(data, timestamps, T, CheckComplete=False)
        _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset_3D(len_closeness=len_closeness, len_period=len_period,
                                                                len_trend=len_trend)
        XC.append(_XC)
        XP.append(_XP)
        XT.append(_XT)
        Y.append(_Y)
        timestamps_Y += _timestamps_Y

    XC = np.vstack(XC)
    XP = np.vstack(XP)
    XT = np.vstack(XT)
    Y = np.vstack(Y)

    XC_train, XP_train, XT_train, Y_train = (XC, XP, XT, Y,)
    XC_test, XP_test, XT_test, Y_test = (XC[-len_test:], XP[-len_test:], XT[-len_test:], Y[-len_test:],)
    X_train = []
    X_test = []
    for l, X_ in zip(
        [len_closeness, len_period, len_trend], [XC_train, XP_train, XT_train]
    ):
        if l > 0:
            X_train.append(X_)
    for l, X_ in zip(
        [len_closeness, len_period, len_trend], [XC_test, XP_test, XT_test]
    ):
        if l > 0:
            X_test.append(X_)

    print("train shape: ", XC_train.shape, Y_train.shape)
    print("test shape: ", XC_test.shape, Y_test.shape)

    if meta_data:
        meta_feature = timestamp2vec(timestamps_Y)
        metadata_dim = meta_feature.shape[1]
        meta_feature_train, meta_feature_test = meta_feature[:-len_test], meta_feature[-len_test:]
        X_train.append(meta_feature_train)
        X_test.append(meta_feature_test)
    else:
        metadata_dim = None

    return (X_train, Y_train, X_test, mmn, metadata_dim)

def sanitize_filename(filename):
    return filename.replace('/', '_').replace('\\', '_').replace(':', '_')

def train_st3dnet(df, grid, len_closeness, len_period, len_trend, T, temporal_granularity='1D'):

    map_height, map_width = grid["ycell"].max() + 1, grid["xcell"].max() + 1
    # map_height, map_width = 24, 16
    num_days = 1
    len_test = T*num_days
    # len_test = 1
    
    X_train, Y_train, X_test, mmn, external_dim = load_data(
        df=df, map_height=map_height, map_width=map_width, T=T, nb_flow=2, 
        len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, 
        len_test=len_test,meta_data=False, temporal_granularity=temporal_granularity
    )

    Y_train = mmn.inverse_transform(Y_train)
    # df = remove_outliers(df)
    result, model = main(X_train, Y_train, X_test, external_dim, grid,len_closeness,len_trend)
    predicts = []

    for i in range(len(result)):
        sum_predict = result[i][0] + result[i][1]
        predicts.append(sum_predict)

    predicts = np.array(predicts)
    
    return result, X_test, model

# file_path = '/home/amarante/Jobs/PM/CrimeBH/IC-CrimeAnalytics/API/data/concatenado1.csv'
# cols_wanted = ["data_hora_inclusao", "natureza_descricao","numero_latitude","numero_longitude"]
# df = pd.read_csv(file_path, usecols=cols_wanted)
# df['data_hora_inclusao'] = pd.to_datetime(df['data_hora_inclusao'])
# df = df.dropna()
# len_closeness = 6
# len_period = 0
# len_trend = 4
# matrix = np.random.rand(0,24,35)

# result, X_test, model = train_st3dnet(df, matrix, len_closeness,len_period,len_trend)