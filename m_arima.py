from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np


# Example function to prepare data
def prepare_data():
    # Example data
    data = [int(x * 3 + 5) for x in range(0, 100)]  # Example time series data
    return data


def predict(coef, history):
    yhat = 0.0
    for i in range(1, len(coef) + 1):
        yhat += coef[i - 1] * history[-i]
    return yhat


def difference(dataset):
    diff = list()
    for i in range(1, len(dataset)):
        value = dataset[i] - dataset[i - 1]
        diff.append(value)
    return np.array(diff)


def predict_arima_verbose(data):
    # Split data into train and test sets
    train = data
    test = data

    history = [x for x in train]
    predictions = list()
    verbose = list()

    for t in range(len(test)):
        model = ARIMA(history, order=(1, 1, 1))
        model_fit = model.fit()
        ar_coef, ma_coef = model_fit.arparams, model_fit.maparams
        resid = model_fit.resid
        diff = difference(history)
        yhat = history[-1] + predict(ar_coef, diff) + predict(ma_coef, resid)
        verbose.append(
            {
                "value": f'{history[-1]} + {predict(ar_coef, diff)} + {predict(ma_coef, resid)}',
                "output": f'{yhat}'
            }
        )
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)

    rmse = sqrt(mean_squared_error(test, predictions))

    return predictions, verbose
