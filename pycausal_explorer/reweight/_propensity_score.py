from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class PropensityScore:
    """
    Implements the Propensity Score model.

    It fits a model with the target being the treatment, where the output is the probability of the individual being
    treated.

    Parameters
    ----------
    model: model used to calculate the propensity score
    scaler: transformation used in the features space
    """

    def __init__(self, model=LogisticRegression(), scaler=StandardScaler()):
        self._estimator_type = model._estimator_type
        self.model = model
        self.scaler = scaler

    def fit(self, X, w):
        X, w = check_X_y(X, w)
        if self.scaler:
            X = self.scaler.fit_transform(X)
        self.model = self.model.fit(X, w)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        if self.scaler:
            X = self.scaler.transform(X)
        return self.model.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        if self.scaler:
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)
