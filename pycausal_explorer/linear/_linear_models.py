from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from pycausal_explorer.base import BaseCausalModel


class CausalLinearRegression(BaseCausalModel):
    def __init__(self, covariates=[], treatment=""):
        self._estimator_type = "regressor"

        if type(covariates) is not list:
            raise ValueError("Covariates must be a list")
        if type(treatment) is not str:
            raise ValueError("Treatment must be a string")

        self.covariates = covariates
        self.treatment = treatment

    def fit(self, X, y):
        X_std = X.copy()
        standard_scaler = StandardScaler()
        X_std[self.covariates] = standard_scaler.fit_transform(X[self.covariates])

        self.standard_scaler = standard_scaler
        self.fitted_model = LinearRegression(n_jobs=-1).fit(
            X_std[[self.treatment] + self.covariates], y
        )
        self.fitted_model_params_ = [
            self.fitted_model.intercept_
        ] + self.fitted_model.coef_.tolist()

    def predict(self, X):
        X_std = X.copy()
        X_std[self.covariates] = self.standard_scaler.fit_transform(X[self.covariates])
        return self.fitted_model.predict(X_std[[self.treatment] + self.covariates])

    def predict_ite(self, X):
        X_std = X.copy()
        X_std[self.covariates] = self.standard_scaler.fit_transform(X[self.covariates])

        X_std_control = X_std.copy()
        X_std_control[self.treatment] = 0

        X_std_treated = X_std.copy()
        X_std_treated[self.treatment] = 1
        return self.fitted_model.predict(
            X_std_treated[[self.treatment] + self.covariates]
        ) - self.fitted_model.predict(X_std_control[[self.treatment] + self.covariates])


class CausalLogisticRegression(BaseCausalModel):
    def __init__(self, covariates=[], treatment=""):
        self._estimator_type = "classifier"

        if type(covariates) is not list:
            raise ValueError("Covariates must be a list")
        if type(treatment) is not str:
            raise ValueError("Treatment must be a string")

        self.covariates = covariates
        self.treatment = treatment

    def fit(self, X, y):
        X_std = X.copy()
        standard_scaler = StandardScaler()
        X_std[self.covariates] = standard_scaler.fit_transform(X[self.covariates])

        self.standard_scaler = standard_scaler
        self.fitted_model = LogisticRegression(
            class_weight="balanced", n_jobs=-1, random_state=42
        ).fit(X_std[[self.treatment] + self.covariates], y)
        self.fitted_model_params_ = [
            self.fitted_model.intercept_
        ] + self.fitted_model.coef_.tolist()

    def predict(self, X):
        X_std = X.copy()
        X_std[self.covariates] = self.standard_scaler.fit_transform(X[self.covariates])
        return self.fitted_model.predict(X_std[[self.treatment] + self.covariates])

    def predict_proba(self, X):
        X_std = X.copy()
        X_std[self.covariates] = self.standard_scaler.fit_transform(X[self.covariates])
        return self.fitted_model.predict_proba(
            X_std[[self.treatment] + self.covariates]
        )[:, 1]

    def predict_ite(self, X):
        X_std = X.copy()
        X_std[self.covariates] = self.standard_scaler.fit_transform(X[self.covariates])

        X_std_control = X_std.copy()
        X_std_control[self.treatment] = 0

        X_std_treated = X_std.copy()
        X_std_treated[self.treatment] = 1
        return (
            self.fitted_model.predict_proba(
                X_std_treated[[self.treatment] + self.covariates]
            )[:, 1]
            - self.fitted_model.predict_proba(
                X_std_control[[self.treatment] + self.covariates]
            )[:, 1]
        )
