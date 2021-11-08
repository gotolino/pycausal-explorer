import numpy as np

from ..base import BaseCausalModel
from ._propensity_score import PropensityScore


class IPTW(BaseCausalModel):
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

    def predict_ite(self, X):
        return np.full(X.shape[0], self._ate)

    def predict_ate(self, X):
        return self._ate
