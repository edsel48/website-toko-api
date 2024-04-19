import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.00001, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Initialize weights and bias
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.num_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            # Compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)
            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return {
            "data": (np.dot(X, self.weights) + self.bias),
            "weights": self.weights,
            "bias": self.bias,
            "algorithm": "Linear Regression"
        }


class MySVR:
    def __init__(self, epsilon=0.1, C=1.0, max_iters=100, learning_rate=0.00001):
        self.epsilon = epsilon
        self.C = C
        self.max_iters = max_iters
        self.learning_rate = learning_rate
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.max_iters):
            for i in range(n_samples):
                # Calculate error for sample i
                error_i = self._predict(X[i]) - y[i]

                # Update weights and bias using the learning rate and gradient
                gradient_w = 0.0 if abs(error_i) < self.epsilon else 2 * self.C * (error_i - np.sign(error_i) * self.epsilon) * X[i]
                gradient_b = 0.0 if abs(error_i) < self.epsilon else 2 * self.C * np.sign(error_i)

                self.w = self.w - self.learning_rate * gradient_w
                self.b = self.b - self.learning_rate * gradient_b

    def _predict(self, X):
        # Calculate the prediction for a single sample
        return np.dot(self.w, X) + self.b

    def predict(self, X):
        # Predict the output for an array of samples
        return {
            "data": (np.array([self._predict(x) for x in X])),
            "weights": self.w,
            "bias": self.b
        }


class MyARIMA:
    def __init__(self, p, d, q, learning_rate=0.01, num_iterations=1000):
        self.p = p  # Autoregressive order
        self.d = d  # Differencing order
        self.q = q  # Moving average order
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.coefficients = None

    def difference(self, data, order):
        return np.diff(data, n=order)

    def fit(self, data):
        # Differencing
        differenced_data = self.difference(data, self.d)

        # Initialize coefficients
        self.coefficients = np.random.randn(self.p + self.q + 1)

        for _ in range(self.num_iterations):
            # Autoregressive part
            ar_part = np.dot(self.coefficients[:self.p], differenced_data[-self.p:][::-1])

            # Moving average part
            ma_part = np.dot(self.coefficients[self.p:self.p + self.q], np.random.randn(self.q))

            # Combined prediction
            prediction = ar_part + ma_part

            # Calculate error
            error = differenced_data[-1] - prediction

            # Update coefficients using gradient descent
            self.coefficients -= self.learning_rate * error * np.concatenate(
                ([1], differenced_data[-self.p:self.p + self.q][::-1]))

    def forecast(self, data, steps):
        forecasted_values = np.zeros(steps)
        forecasted_data = np.zeros(steps)
        differenced_data = self.difference(data, self.d)

        for i in range(steps):
            # Autoregressive part
            ar_part = np.dot(self.coefficients[:self.p], differenced_data[-self.p:][::-1])

            # Moving average part
            ma_part = np.dot(self.coefficients[self.p:self.p + self.q], np.random.randn(self.q))

            # Combined prediction
            prediction = ar_part + ma_part

            # Invert differencing
            forecasted_values[i] = data[-1] + prediction

            # forecasted_data[i] = {
            #     "data": data[-1],
            #     "AR": ar_part,
            #     "MA": ma_part,
            #     "prediction": prediction
            # }

            # Update data for the next iteration
            differenced_data = np.concatenate((differenced_data, [prediction]))

        return {
            "data": forecasted_values,
            # "prediction_steps": forecasted_data,
            "algorithm": "ARIMA"
        }
