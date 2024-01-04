import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
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
        return np.dot(X, self.weights) + self.bias


class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            # Hinge loss and gradient calculation
            margin = 1 - y * (np.dot(X, self.weights) + self.bias)
            dw = np.zeros(num_features)
            db = 0

            for i in range(num_samples):
                if margin[i] > 0:
                    dw -= y[i] * X[i]
                    db -= y[i]

            dw = dw / num_samples + 2 * self.lambda_param * self.weights
            db = db / num_samples

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.sign(np.dot(X, self.weights) + self.bias)


class myARIMA:
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

            # Update data for the next iteration
            differenced_data = np.concatenate((differenced_data, [prediction]))

        return forecasted_values
