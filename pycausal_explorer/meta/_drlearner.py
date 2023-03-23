import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted, check_X_y

from pycausal_explorer.base import BaseCausalModel

from ..reweight import PropensityScore


class DRLearner(BaseCausalModel):
    """
    Implementation of the Doubly Roobust Learner.

    It consists of estimating heterogeneous treatment effect doubly robust.
    Details of DR-learner theory are available at Edward H. Kennedy (2020) (https://arxiv.org/abs/2004.14497).

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
            self.u0 = [clone(learner), clone(learner)]
            self.u1 = [clone(learner), clone(learner)]
            self.tau = [clone(learner), clone(learner)]
        elif learner is None and all([model is not None for model in [u0, u1, tau]]):
            self.u0 = [clone(u0), clone(u0)]
            self.u1 = [clone(u1), clone(u1)]
            self.tau = [clone(tau), clone(tau)]
        else:
            raise ValueError("Either learner or (u0, u1, tau) must be passed")
        self.propensity_score = [PropensityScore(), PropensityScore()]
        self.random_state = random_state

    def fit(self, X, y, *, treatment):
        X, y = check_X_y(X, y)
        X, w = check_X_y(X, treatment)
        X1_split, X2_split, y1_split, y2_split, w1_split, w2_split = train_test_split(
            X, y, w, test_size=0.5, random_state=42
        )
        X_train = [X1_split, X2_split]
        y_train = [y1_split, y2_split]
        w_train = [w1_split, w2_split]

        for i in range(0, 2):
            X1, y1, w1 = X_train[i], y_train[i], w_train[i]
            X2, y2, w2 = X_train[1 - i], y_train[1 - i], w_train[1 - i]

            self.g[i].fit(X1, w1)

            X1_treat = X1[w == 1].copy()
            X1_control = X1[w == 0].copy()

            y1_1 = y1[w == 1].copy()
            y1_0 = y1[w == 0].copy()

            self.u0[i] = self.u0[i].fit(X1_control, y1_0)
            self.u1[i] = self.u1[i].fit(X1_treat, y1_1)

            uw = np.empty(shape=[X2.shape[0], 1])
            if 1 in w:
                uw[w == 1] = self.u1[i].predict(X1[w == 1]).reshape(-1, 1)
            if 0 in w:
                uw[w == 0] = self.u0[i].predict(X1[w == 0]).reshape(-1, 1)

            g_proba = self.g[i].predict_proba(X2)[:, 1]
            pseudo_outcomes = (
                (w2 - g_proba) / ((g_proba) * (1 - g_proba)) * (y2 - uw)
                + self.u1[i].predict(X2)
                - self.u0[i].predict(X2)
            )

            self.tau[i] = self.tau[i].fit(X2, pseudo_outcomes)

        self.is_fitted_ = True
        return self

    def predict_ite(self, X):
        predictions = (self.tau[0].predict_proba(X) + self.tau[1].predict_proba(X)) / 2
        return predictions
