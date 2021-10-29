from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


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
        self.model = model
        self.scaler = scaler

    def fit(self, X, w):
        X_scaled = self.scaler.fit_transform(X)
        self.model = self.model.fit(X_scaled, w)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
