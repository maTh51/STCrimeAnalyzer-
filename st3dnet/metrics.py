import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import time

class MinMaxNormalization(object):
    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X

def mean_squared_error(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

def mae(actual, predicted):
    mae = tf.reduce_mean(tf.abs(actual - predicted))
    return mae

def ALS(y_true, y_pred):
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    log_y_pred = K.log(y_pred)
    ALS = -K.mean(K.sum(y_true * log_y_pred, axis=-1))
    return ALS

def mape(actual, predicted):
    mape = tf.reduce_mean(tf.abs((actual - predicted) / (actual))) * 100
    return mape

def absolute(actual):
    abs = tf.reduce_sum(actual)
    return abs

def relative(actual, predicted):
    total_actual = tf.reduce_sum(actual)
    total_predicted = tf.reduce_sum(predicted)
    rel = total_predicted / total_actual
    return rel