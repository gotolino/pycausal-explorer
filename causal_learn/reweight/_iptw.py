import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ._propensity_score import PropensityScore


class IPTW(BaseEstimator):
    """
    Implements Inverse Probability Treatment Weighting (IPTW) model.

    This model weights the units based on the propensity score. It uses all units to get the
    average treatment effect, so all units receives the same ite.
    Parameters
    ----------
    propensity_score: model used to calculate propensity score.
    """

    def __init__(self, propensity_score=PropensityScore()):
        self.propensity_score = propensity_score
        self.weight_ = None

    def fit(self, X, w, y):
        self.propensity_score.fit(X, w)
        propensity_score_hat = self.propensity_score.predict_proba(X)[:, 1]
        self.weight_ = (w - propensity_score_hat) / (
            propensity_score_hat * (1 - propensity_score_hat)
        )
        self._ate = np.mean(self.weight_ * y)

    def predict(self, X, w):
        pass

    def predict_ate(self, X):
        return self._ate
