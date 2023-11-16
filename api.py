# FLASK FOR API PURPOSES
from flask import Flask, jsonify, request

# DATA PROCESSING STUFF
import numpy as np
import pandas as pd
import json

# ARIMA STUFF
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

# LINEAR REGRESSION
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# SVR
from sklearn.svm import SVR


def prepare_dataframe(data):
    training_index = range(len(data))
    col = ['sold']

    return pd.DataFrame(data=data,
                        index=training_index,
                        columns=col)


def get_order(dataframe):
    stepwise_fit = auto_arima(dataframe['sold'], trace=True, suppress_warnings=True)
    order = stepwise_fit.to_dict()['order']

    return order


def arima_model(data, order):
    model = ARIMA(data['sold'], order=order)
    model = model.fit()

    model.summary()

    return model


def process_data(training_data):
    # done
    #   1. get the training data and swap it into a data frame v
    #   2. get the order for the best training setup for arima
    #   3. getting the training data ready for action
    #   4. train the data
    #   5. prepare the prediction to an array for output
    #   6. output the data

    training_dataframe = prepare_dataframe(training_data)
    order = get_order(training_dataframe)

    model = arima_model(training_dataframe, order)

    return model


def parser_helper(request):
    # getting the data from request
    record = json.loads(request.data)

    # preparing the data so it now when to start and end
    start = int(record['start'])
    end = int(record['end'])

    sold_data = [int(x) for x in record['sold_data']]

    return start, end, sold_data


def prepare_data(start, end, sold_data):
    _start = start - 1
    _end = end

    # Preparing array that will be used
    X = []
    y = []

    # looping through every single data
    for index, data in enumerate(sold_data):
        X.append(index)
        y.append(data)

    # Cutting the array into pieces
    X = np.array([[x] for x in X[_start: _end]])
    y = np.array(y[_start: _end])

    return X, y


def get_lr_model(start, end, sold_data):
    X, y = prepare_data(start, end, sold_data)

    reg = linear_model.LinearRegression()
    reg.fit(X, y)

    return reg


def get_svr_model(start, end, sold_data):
    X, y = prepare_data(start, end, sold_data)

    svr = SVR(kernel='linear')
    svr.fit(X, y)

    return svr


app = Flask(__name__)


@app.route('/')
def home():
    data = "TEST DATA CHECK"
    return jsonify({'data': data})


@app.route('/predict/arima', methods=['POST'])
def predict_arima():
    start, end, sold_data = parser_helper(request)

    # creating the model
    model = process_data(sold_data)

    # predicting the data from the model
    predict = model.predict(start=start-1, end=end-1, typ='levels')

    # changing the type from a numpy list into a normal list
    prediction = list(predict.apply(np.round))

    return jsonify({
        "start": start,
        "end": end,
        "predicted": prediction,
    })

@app.route('/predict/linear-regression', methods=['POST'])
def predict_linear_regression():
    start, end, sold_data = parser_helper(request)
    lr_model = get_lr_model(start, end, sold_data)

    predict = lr_model.predict([[x] for x in range(start-1, end)])
    print(type(predict))
    predicted = [int(x) for x in list(predict)]

    return jsonify({
        "start": start,
        "end": end,
        "predicted": predicted,
    })


@app.route("/predict/svr", methods=["POST"])
def predict_svr():
    start, end, sold_data = parser_helper(request)
    svr_model = get_svr_model(start, end, sold_data)

    predict = svr_model.predict([[x] for x in range(start-1, end)])

    predicted = [int(x) for x in list(predict)]

    return jsonify({
        "start": start,
        "end": end,
        "predicted": predicted,
    })


if __name__ == '__main__':
    app.run(debug=True, port=7373)
