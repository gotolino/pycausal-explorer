import numpy as np
from sklearn.metrics import mean_squared_error


def pehe(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))
