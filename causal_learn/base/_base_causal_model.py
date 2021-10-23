from sklearn.base import BaseEstimator


class BaseCausalModel(BaseEstimator):
    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def predict_ite(self, X):
        pass
