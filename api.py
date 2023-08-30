from flask import Flask, jsonify, request
import random as rnd

app = Flask(__name__)

def random_function():
    return rnd.randint(1, 5)


@app.route('/')
def home():
    data = "HELLO GAES"
    return jsonify({'data': data})


@app.route("/test")
def test():
    data = random_function()
    return jsonify({'data': data})


if __name__ == '__main__':
    app.run(debug=True)
