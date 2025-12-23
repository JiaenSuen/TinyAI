import numpy as np
from sklearn.model_selection import train_test_split

class NaiveBayes:
    def __init__(self):
        self.eps = 1e-6

    def fit(self, X, y):
        self.num_examples, self.num_features = X.shape
        self.num_classes = len(np.unique(y))
        self.classes_mean = {}
        self.classes_variance = {}
        self.classes_prior = {}

        for c in range(self.num_classes):
            X_c = X[y == c]
            self.classes_mean[c] = np.mean(X_c, axis=0)
            self.classes_variance[c] = np.var(X_c, axis=0)
            self.classes_prior[c] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        num_examples = X.shape[0]
        probs = np.zeros((num_examples, self.num_classes))

        for c in range(self.num_classes):
            prior = np.log(self.classes_prior[c])
            probs_c = self._density_function(X, self.classes_mean[c], self.classes_variance[c])
            probs[:, c] = probs_c + prior

        return np.argmax(probs, axis=1)

    def _density_function(self, x, mean, sigma):
        const = -0.5 * np.sum(np.log(2 * np.pi * (sigma + self.eps)))
        probs = -0.5 * np.sum(((x - mean) ** 2) / (sigma + self.eps), axis=1)
        return const + probs

 