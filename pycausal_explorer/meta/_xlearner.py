import numpy as np

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted, check_X_y

from pycausal_explorer.base import BaseCausalModel

from ..reweight import PropensityScore


class XLearner(BaseCausalModel):
    """
    Implementation of the X-learner.

    It consists of estimating heterogeneous treatment effect using four machine learning models.
    Details of X-learner theory are available at Kunzel et al. (2018) (https://arxiv.org/abs/1706.03461).

    Parameters
    ----------
    learner: base learner to use in all models. Either leaner or (u0, u1, te_u0, te_u1) must be filled
    u0: model used to estimate outcome in the control group
    u1: model used to estimate outcome in the treatment group
    te_u0: model used to estimate treatment effect in the control group
    te_u1: model used to estimate treatment effect in the treatment group group
    random_state: random state
    """

    def __init__(
        self,
        learner=RandomForestRegressor(),
        u0=None,
        u1=None,
        te_u0=None,
        te_u1=None,
        random_state=42,
    ):
        self.learner = learner
        if learner is not None and all(
            [model is None for model in [u0, u1, te_u0, te_u1]]
        ):
            self.u0 = clone(learner)
            self.u1 = clone(learner)
            self.te_u0 = clone(learner)
            self.te_u1 = clone(learner)
        elif learner is None and all(
            [model is not None for model in [u0, u1, te_u0, te_u1]]
        ):
            self.u0 = clone(u0)
            self.u1 = clone(u1)
            self.te_u0 = clone(te_u0)
            self.te_u1 = clone(te_u1)
        else:
            raise ValueError("Either learner or (u0, u1, te_u0, te_u1) must be passed")

        self._estimator_type = self.u0._estimator_type
        self.g = PropensityScore()
        self.random_state = random_state

    def fit(self, X, y, *, treatment):
        X, y = check_X_y(X, y)
        X, w = check_X_y(X, treatment)
        self.g.fit(X, w)

        X_treat = X[w == 1].copy()
        X_control = X[w == 0].copy()

        y1 = y[w == 1].copy()
        y0 = y[w == 0].copy()

        self.u0 = self.u0.fit(X_control, y0)
        self.u1 = self.u1.fit(X_treat, y1)

        y1_pred = self.u1.predict(X_control)
        y0_pred = self.u0.predict(X_treat)
        te_imp_control = y1_pred - y0
        te_imp_treat = y1 - y0_pred

        self.te_u0 = self.te_u0.fit(X_control, te_imp_control)
        self.te_u1 = self.te_u1.fit(X_treat, te_imp_treat)

        self.is_fitted_ = True
        return self

    def predict(self, X, w):
        check_is_fitted(self)
        predictions = np.empty(shape=[X.shape[0], 1])

        if 1 in w:
            predictions[w == 1] = self.u1.predict(X[w == 1]).reshape(-1, 1)
        if 0 in w:
            predictions[w == 0] = self.u0.predict(X[w == 0]).reshape(-1, 1)
        return predictions

    def predict_ite(self, X):
        check_is_fitted(self)
        g_x = self.g.predict_proba(X)[:, 1]
        result = g_x * self.te_u0.predict(X) + (1 - g_x) * self.te_u1.predict(X)
        return result
