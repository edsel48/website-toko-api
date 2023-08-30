from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/')
def home():
    data = "HELLO GAES"
    return jsonify({'data': data})


if __name__ == '__main__':
    app.run(debug=True)
