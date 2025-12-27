from sklearn.metrics import r2_score


class Residual_Stack:
    def __init__(self,  base_model=None, residual_model=None):
        self.base_model = base_model
        self.residual_model = residual_model
        self.is_fitted = False

    def fit(self, X, y):
        # Train Base Layer
        self.base_model.fit(X, y)
        base_pred = self.base_model.predict(X)

        # Train Residual Model
        residual = y - base_pred
        self.residual_model.fit(X, residual)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model haven't trained...")
        base_pred = self.base_model.predict(X)
        residual_pred = self.residual_model.predict(X)
        return base_pred + residual_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)