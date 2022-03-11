import numpy as np
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.utils.validation import check_is_fitted

from pycausal_explorer.base import BaseCausalModel

from sklearn.base import clone


class DoubleMLLinear(BaseCausalModel):
    """
    Double Machine Learning model. Estimates causal effect using two different models:
    one models outcome, and another models treatment.

    Linear version. Should be used when you believe treatment effect is linear, and
    the treatment variable is continuous

    Parameters
    ----------
    outcome_learner : estimator object
        base learner to use when predicting outcome. Should implement fit and predict methods.
    treatment_learner : estimator object
        base learner to use when predicting treatment. Should implement fit and predict methods.
    score : basestring
        Which score function to use. One of "partial-out" and "orthogonal"
    """

    def __init__(
        self, outcome_learner, treatment_learner, score="partial-out", k_fold=5
    ):
        valid_scores = ["partial-out", "orthogonal"]
        if score not in valid_scores:
            raise ValueError("Score has to be one of " + str(valid_scores))
        self.outcome_leaner = clone(outcome_learner)
        self.treatment_learner = clone(treatment_learner)

        self.k_fold = k_fold
        self.score = score
        self.is_fitted_ = False

    def fit(self, X, y, *, treatment):
        self.is_fitted_ = True
        pred_outcome = cross_val_predict(self.outcome_leaner, X, y, cv=self.k_fold)
        pred_treatment = cross_val_predict(
            self.treatment_learner, X, treatment, cv=self.k_fold
        )

        if self.score == "partial-out":
            self._psi_a = (-treatment * (treatment - pred_treatment)).mean()
        elif self.score == "orthogonal":
            self._psi_a = (-np.square(treatment - pred_treatment)).mean()
        self._psi_b = ((y - pred_outcome) * (treatment - pred_treatment)).mean()

    def predict_ite(self, X):
        check_is_fitted(self)
        return np.full(X.shape[0], -self._psi_b / self._psi_a)


class DoubleMLBinaryTreatment(BaseCausalModel):
    """
    Double Machine Learning model. Estimates causal effect using two different models:
    one models outcome, and another models treatment.

    Binary treatment version. Should be used when treatment is a binary variable.

    Parameters
    ----------
    outcome_learner : estimator object
        base learner to use when predicting outcome. Should implement fit and predict methods.
    treatment_learner : estimator object
        base learner to use when predicting treatment probability. Should be a classifier
        learner that implements fit and predict_proba methods.
    """

    def __init__(self, outcome_learner, treatment_learner):
        self.outcome_leaner = clone(outcome_learner)
        self.treatment_learner = clone(treatment_learner)

        self.is_fitted_ = False

    def fit(self, X, y, *, treatment):
        self.is_fitted_ = True

        X_t, X_r, y_t, y_r, w_t, w_r = train_test_split(X, y, treatment, train_size=0.5)
        reg_size = X_r.shape[0]

        self.outcome_leaner.fit(np.column_stack([w_t, X_t]), y_t)
        pred_y_treat = self.outcome_leaner.predict(
            np.column_stack([np.ones(reg_size), X_r])
        )
        pred_y_cont = self.outcome_leaner.predict(
            np.column_stack([np.zeros(reg_size), X_r])
        )

        self.treatment_learner.fit(X_t, w_t)
        pred_w = self.treatment_learner.predict_proba(X_r)[:, 1]

        self._ate = (
            pred_y_treat
            - pred_y_cont
            + w_r * (y_r - pred_y_treat) / pred_w
            - (np.ones(reg_size) - w_r)
            * (y_r - pred_y_cont)
            / (np.ones(reg_size) - pred_w)
        ).mean()

    def predict_ite(self, X):
        check_is_fitted(self)

        return np.full(X.shape[0], self._ate)
