import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted, check_X_y

from pycausal_explorer.base import BaseCausalModel

from ..reweight import PropensityScore


class PWLearner(BaseCausalModel):
    """
    Implementation of the Regression Adjustment Learner.

    It consists of estimating heterogeneous treatment effect doubly robust.
    Details of RA-learner theory are available at Edward H. Kennedy (2020) (https://arxiv.org/abs/2004.14497).

    Parameters
    ----------
    learner: base learner to use in all models. Either leaner or (u0, u1, tau) must be filled
    u0: model used to estimate outcome in the control group
    u1: model used to estimate outcome in the treatment group
    tau: model used to estimate treatment effect of pseudo-outcome
    random_state: random state
    """

    def __init__(self, learner=None, u0=None, u1=None, tau=None, random_state=42):
        self.learner = learner
        if learner is not None and all([model is None for model in [u0, u1, tau]]):
            self.u0 = clone(learner)
            self.u1 = clone(learner)
            self.tau = clone(learner)
        elif learner is None and all([model is not None for model in [u0, u1, tau]]):
            self.u0 = clone(u0)
            self.u1 = clone(u1)
            self.tau = clone(tau)
        else:
            raise ValueError("Either learner or (u0, u1, tau) must be passed")
        self.propensity_score = PropensityScore()
        self.random_state = random_state

    def fit(self, X, y, *, treatment):
        X, y = check_X_y(X, y)
        X, w = check_X_y(X, treatment)

        self.propensity_score.fit(X, w)

        X_treat = X[w == 1].copy()
        X_control = X[w == 0].copy()

        y1 = y[w == 1].copy()
        y0 = y[w == 0].copy()

        self.u0 = self.u0.fit(X_control, y0)
        self.u1 = self.u1.fit(X_treat, y1)

        propensity_score_proba = np.maximum(
            self.propensity_score.predict_proba(X)[:, 1], 0.001
        )
        propensity_score_proba = np.minimum(propensity_score_proba, 0.999)

        y_pseudo = (
            w / propensity_score_proba - (1 - w) / (1 - propensity_score_proba)
        ) * y

        self.tau.fit(X, y_pseudo)

        self.is_fitted_ = True
        return self

    def predict_ite(self, X):
        check_is_fitted(self)
        predictions = self.tau.predict(X)
        return predictions
