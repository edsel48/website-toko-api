# FLASK FOR API PURPOSES
from flask import Flask, jsonify, request

# DATA PROCESSING STUFF
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

# ARIMA STUFF
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA


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


app = Flask(__name__)


@app.route('/')
def home():
    data = "TEST DATA CHECK"
    return jsonify({'data': data})


@app.route('/', methods=['POST'])
def testing():
    # getting the data from request
    record = json.loads(request.data)

    # preparing the data so it now when to start and end
    start = int(record['start'])
    end = int(record['end'])

    # creating the model
    model = process_data(record['sold_data'])

    # predicting the data from the model
    predict = model.predict(start=start, end=end, typ='levels')

    # changing the type from a numpy list into a normal list
    prediction = list(predict.apply(np.round))

    return jsonify({
        "start": start,
        "end": end,
        "predicted": prediction,
    })


if __name__ == '__main__':
    app.run(debug=True)
