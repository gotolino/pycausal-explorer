import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, check_X_y

from pycausal_explorer.base import BaseCausalModel
from pycausal_explorer.reweight import PropensityScore
from pycausal_explorer.nearest_neighbors import CausalKNNRegressor


class PSM(BaseCausalModel):
    def __init__(
        self,
        propensity_score=PropensityScore(),
        matching_model=CausalKNNRegressor(),
    ):
        # if type(params) is not dict:
        #     raise ValueError("KNN params must be a dictionary")
        # if type(scale) is not bool:
        #     raise ValueError("Scale flag must be a Boolean value")

        self.propensity_score = propensity_score
        self.matching_model = matching_model


    def fit(self, X, y, *, treatment):
        X, y = check_X_y(X, y)
        X, w = check_X_y(X, treatment)
        self.propensity_score.fit(X, w)

        X_propensity = self.propensity_score.predict_proba(X)

        self.matching_model.fit(X_propensity, y, treatment=w)

        self.is_fitted_ = True
        return self

    def predict(self, X, w):
        check_is_fitted(self)
        X = self.propensity_score.predict_proba(X)
        return self.matching_model.predict(X, w)

    def predict_ite(self, X):
        check_is_fitted(self)
        X = self.propensity_score.predict_proba(X)
        return self.matching_model.predict_ite(X)
