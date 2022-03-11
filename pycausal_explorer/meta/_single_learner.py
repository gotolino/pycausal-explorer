import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted, check_X_y

from pycausal_explorer.base import BaseCausalModel


class SingleLearnerBase(BaseCausalModel):
    def __init__(self, learner):
        if isinstance(learner, type):
            raise ValueError(
                "You should provide an instance of an estimator instead of a class."
            )
        else:
            self.learner = clone(learner)

    def fit(self, X, y, *, treatment):
        X, y = check_X_y(X, y)
        X, w = check_X_y(X, treatment)
        self.learner = self.learner.fit(np.column_stack([X, w]), y)
        self.is_fitted_ = True
        return self

    def predict(self, X, w):
        check_is_fitted(self)
        return self.learner.predict(np.column_stack([X, w]))


class SingleLearnerRegressor(SingleLearnerBase):
    """
    Implementation of the single learner model.

    Regressor version, should be used with continuous data.

    Uses a single provided model to predict outcome when under treatment, and when not.
    Uses that to estimate treatment effect.

    Parameters
    ----------
    learner: estimator object
        base learner to use when predicting outcome. Should implement fit and predict methods.
    """

    def __init__(self, learner):
        super().__init__(learner)
        self._estimator_type = "regressor"

    def predict_ite(self, X):
        check_is_fitted(self)
        return self.predict(X, np.ones(shape=X.shape[0])) - self.predict(
            X, np.zeros(shape=X.shape[0])
        )


class SingleLearnerClassifier(SingleLearnerBase):
    """
    Implementation of the single learner model.

    Logistic version, should be used with binary data.

    Uses a single provided model to predict outcome when under treatment, and when not.
    Uses that to estimate treatment effect.

    Parameters
    ----------
    learner: estimator object
        base learner to use when predicting outcome. Should implement fit and predict methods.
    """

    def __init__(self, learner):
        super().__init__(learner)
        self._estimator_type = "classifier"

    def predict_proba(self, X, w):
        return self.learner.predict_proba(np.column_stack([X, w]))

    def predict_ite(self, X):
        check_is_fitted(self)
        return (
            self.predict_proba(X, np.ones(shape=X.shape[0]))[:, 1]
            - self.predict_proba(X, np.zeros(shape=X.shape[0]))[:, 1]
        )
