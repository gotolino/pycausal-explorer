import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, check_X_y

from pycausal_explorer.base import BaseCausalModel


class CausalLinearRegression(BaseCausalModel):
    """
    Causal Linear Regressor model.

    Estimates causal effect using a Linear Regressor.
    """

    def __init__(self):
        self._estimator_type = "regressor"
        self.standard_scaler = StandardScaler()

    def fit(self, X, y, *, treatment):
        X, y = check_X_y(X, y)
        X, w = check_X_y(X, treatment)
        X_std = self.standard_scaler.fit_transform(X)
        self.fitted_model = LinearRegression(n_jobs=-1).fit(
            np.column_stack([X_std, w]), y
        )
        self.fitted_model_params_ = [
            self.fitted_model.intercept_
        ] + self.fitted_model.coef_.tolist()
        self.is_fitted_ = True
        return self

    def predict(self, X, w):
        check_is_fitted(self)
        X_std = self.standard_scaler.fit_transform(X)
        return self.fitted_model.predict(np.column_stack([X_std, w]))

    def predict_ite(self, X):
        check_is_fitted(self)
        X_std = self.standard_scaler.fit_transform(X)

        return self.fitted_model.predict(
            np.column_stack([X_std, np.ones(shape=(X_std.shape[0], 1))])
        ) - self.fitted_model.predict(
            np.column_stack([X_std, np.zeros(shape=(X_std.shape[0], 1))])
        )


class CausalLogisticRegression(BaseCausalModel):
    """
    Causal Logistic Regressor model.

    Estimates causal effect using a Logistic Regressor.
    """

    def __init__(self):
        self._estimator_type = "classifier"
        self.standard_scaler = StandardScaler()

    def fit(self, X, y, *, treatment):
        X, y = check_X_y(X, y)
        X, w = check_X_y(X, treatment)
        X_std = self.standard_scaler.fit_transform(X)

        self.fitted_model = LogisticRegression(
            class_weight="balanced", n_jobs=-1, random_state=42
        ).fit(np.column_stack([X_std, w]), y)
        self.fitted_model_params_ = [
            self.fitted_model.intercept_
        ] + self.fitted_model.coef_.tolist()
        self.is_fitted_ = True
        return self

    def predict(self, X, w):
        check_is_fitted(self)
        X_std = self.standard_scaler.fit_transform(X)
        return self.fitted_model.predict(np.column_stack([X_std, w]))

    def predict_proba(self, X, w):
        check_is_fitted(self)
        X_std = self.standard_scaler.fit_transform(X)
        return self.fitted_model.predict_proba(np.column_stack([X_std, w]))[:, 1]

    def predict_ite(self, X):
        check_is_fitted(self)
        X_std = self.standard_scaler.fit_transform(X)
        return (
            self.fitted_model.predict_proba(
                np.column_stack([X_std, np.ones(shape=(X_std.shape[0], 1))])
            )[:, 1]
            - self.fitted_model.predict_proba(
                np.column_stack([X_std, np.zeros(shape=(X_std.shape[0], 1))])
            )[:, 1]
        )
