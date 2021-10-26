import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ._propensity_score import PropensityScore


class IPTW(BaseEstimator):
    def __init__(self, propensity_score=PropensityScore()):
        self.propensity_score = propensity_score
        self._weight = None

    def fit(self, X, w, y):
        self.propensity_score.fit(X, w)
        propensity_score_hat = self.propensity_score.predict_proba(X)[:, 1]
        self._weight = (w - propensity_score_hat) / (
            propensity_score_hat * (1 - propensity_score_hat)
        )
        self._ate = np.mean(self._weight * y)

    def predict(self, X, w):
        pass

    def predict_ate(self, X):
        return self._ate
