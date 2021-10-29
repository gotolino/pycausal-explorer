import numpy as np

from sklearn.base import BaseEstimator


class BaseCausalModel(BaseEstimator):
    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def predict_ite(self, X):
        pass

    def predict_ate(self, X):
        return np.mean(self.predict_ite(X))
