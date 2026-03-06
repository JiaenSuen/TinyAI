import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from scipy.special import softmax

class Residual_Stack_Classifier:
 
    def __init__(self, base_model=None, residual_model=None):
        self.base_model = base_model
        self.residual_model = residual_model
        self.encoder = OneHotEncoder(sparse_output=False)
        self.is_fitted = False

    def _to_logit(self, prob):
        #to logit 
        eps = 1e-7
        prob = np.clip(prob, eps, 1 - eps)
        return np.log(prob)

    def fit(self, X, y):
        y = np.array(y).reshape(-1, 1)
        # one-hot label
        y_onehot = self.encoder.fit_transform(y)

        # train base model
        self.base_model.fit(X, y.ravel())
        base_prob = self.base_model.predict_proba(X)
        base_logit = self._to_logit(base_prob)

        # compute logit residual
        y_logit = self._to_logit(y_onehot)
        residual = y_logit - base_logit

        # train residual model
        self.residual_model.fit(X, residual)

        self.is_fitted = True
        return self

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model hasn't been trained yet.")

        base_prob = self.base_model.predict_proba(X)
        base_logit = self._to_logit(base_prob)

        residual_pred = self.residual_model.predict(X)

        # add residual in logit space
        final_logit = base_logit + residual_pred

        # back to probability
        final_prob = softmax(final_logit, axis=1)
        return final_prob

    def predict(self, X):
        prob = self.predict_proba(X)
        return np.argmax(prob, axis=1)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
