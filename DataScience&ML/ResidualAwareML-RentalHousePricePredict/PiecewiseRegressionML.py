import pandas as pd
from sklearn.metrics import r2_score



class SegmentedModel:
    def __init__(self, split_col, split_value=None, model_small=None, model_large=None):
        self.split_col = split_col
        self.split_value = split_value
        self.model_small = model_small
        self.model_large = model_large
        self.is_fitted = False

    def fit(self, X, y):
        # Automatic decision split_value
        if self.split_value is None:
            self.split_value = X[self.split_col].median()

        # Segmentation training materials
        X_small = X[X[self.split_col] <= self.split_value]
        y_small = y.loc[X_small.index]
        X_large = X[X[self.split_col] > self.split_value]
        y_large = y.loc[X_large.index]

        # Training interval model
        if self.model_small is not None and not X_small.empty:
            self.model_small.fit(X_small, y_small)
        if self.model_large is not None and not X_large.empty:
            self.model_large.fit(X_large, y_large)

        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model haven't trained...")

        preds = pd.Series(index=X.index, dtype=float)

        # Small interval prediction
        if self.model_small is not None:
            X_small = X[X[self.split_col] <= self.split_value]
            if not X_small.empty:
                preds.loc[X_small.index] = self.model_small.predict(X_small)

        # Large-range forecast
        if self.model_large is not None:
            X_large = X[X[self.split_col] > self.split_value]
            if not X_large.empty:
                preds.loc[X_large.index] = self.model_large.predict(X_large)

        return preds.values

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
