from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class IPTW(BaseEstimator):
    def __init__(self, propensity_score_model=LogisticRegression):
        self.propensity_score = propensity_score_model
        self.standard_scaler = StandardScaler()

    def fit(self, X, w, y):
        pass

    def _fit_propensity_score(self, X, w):
        X_scaled = self.standard_scaler.fit_transform(X)
        self.g = self.g.fit(X_scaled, w)

    def predict(self, X, w):
        pass

    def predict_ate(self, X):
        pass
