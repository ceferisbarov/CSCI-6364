from data import train_X, val_X, test_X, train_y, val_y, test_y
import numpy as np
from tqdm import tqdm

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegressionSGD:
    def __init__(self, learning_rate=0.01, n_iters=1000, l2_penalty=0, progress_steps=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.l2_penalty = l2_penalty
        self.progress_steps = progress_steps

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for _ in range(self.n_iters):
            pbar = tqdm(total=n_samples, desc="Processing")
            for i in range(n_samples):
                linear_model = np.dot(X[i], self.weights)
                # y_predicted = sigmoid(linear_model)

                power = -y[i] * linear_model
                a = y[i] * np.exp(power)
                b = 1 + np.exp(power)
                dw = (a / b) * X[i].T + self.l2_penalty * self.weights
                # dw = (y_predicted - y[i]) * X[i] + self.l2_penalty * self.weights

                self.weights -= self.learning_rate * dw
                if i %  self.progress_steps == 0:
                    train_acc = self.accuracy(train_X, train_y)
                    validate_acc = self.accuracy(val_X, val_y)
                    pbar.set_description(f"(train acc: {train_acc:.2f}), (val acc: {validate_acc:.2f})")
                    pbar.update(self.progress_steps)
                    pbar.refresh()

    def predict(self, X):
        linear_model = np.dot(X, self.weights)
        y_predicted = sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)
    
    def accuracy(self, X, y):
        predicted_labels = self.predict(X)
        return np.mean(y == predicted_labels)

if __name__ == "__main__":
    model = LogisticRegressionSGD(learning_rate=0.1, n_iters=1, l2_penalty=2, progress_steps=5000)
    model.fit(train_X, train_y)
    print("test accuracy: ", model.accuracy(test_X, test_y))
