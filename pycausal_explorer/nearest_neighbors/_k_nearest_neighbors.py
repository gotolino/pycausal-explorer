import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, check_X_y

from pycausal_explorer.base import BaseCausalModel


class CausalKNNBaseModel(BaseCausalModel):
    def __init__(
        self,
        params={"n_neighbors": 10, "metric": "euclidean"},
        scale=True,
    ):
        if type(params) is not dict:
            raise ValueError("KNN params must be a dictionary")
        if type(scale) is not bool:
            raise ValueError("Scale flag must be a Boolean value")

        self.params = params
        self.scale = scale

        self.standard_scaler = StandardScaler()

    def _scale_input_data_if_indicated(self, X):
        if self.scale:
            X_array = self.standard_scaler.fit_transform(X)
            X = X_array
        return X

    def fit(self, X, y, *, treatment):
        X, y = check_X_y(X, y)
        X, w = check_X_y(X, treatment)
        X = self._scale_input_data_if_indicated(X)

        X_treat = X[w == 1].copy()
        X_control = X[w == 0].copy()

        y_treat = y[w == 1].copy()
        y_control = y[w == 0].copy()

        # Train KNN model for the control group
        self.knn_control_model.fit(
            X=X_control,
            y=y_control,
        )

        # Train KNN model for the treated samples
        self.knn_treated_model.fit(
            X=X_treat,
            y=y_treat,
        )

        self.is_fitted_ = True
        return self

    def predict(self, X, w):
        check_is_fitted(self)
        X = self._scale_input_data_if_indicated(X)

        y_predict_0 = self.knn_control_model.predict(X)
        y_predict_1 = self.knn_treated_model.predict(X)

        return np.where(w == 1, y_predict_1, y_predict_0)

    def predict_ite(self, X):
        check_is_fitted(self)
        X = self._scale_input_data_if_indicated(X)

        try:
            # Predict y0 for the test set
            y_predict_0 = self.knn_control_model.predict(X)

            # Predict y1 for the test set
            y_predict_1 = self.knn_treated_model.predict(X)
        except IndexError:
            print(
                "Positivity has been violated: either control or treatment group has only one y class."
            )
            raise

        return y_predict_1 - y_predict_0


class CausalKNNRegressor(CausalKNNBaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._estimator_type = "regressor"
        self.knn_control_model = KNeighborsRegressor(**self.params)
        self.knn_treated_model = KNeighborsRegressor(**self.params)


class CausalKNNClassifier(CausalKNNBaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._estimator_type = "classifier"
        self.knn_control_model = KNeighborsClassifier(**self.params)
        self.knn_treated_model = KNeighborsClassifier(**self.params)

    def predict_proba(self, X, w):
        check_is_fitted(self)
        self._scale_input_data_if_indicated(X)

        y_predict_0 = self.knn_control_model.predict_proba(X)
        y_predict_1 = self.knn_treated_model.predict_proba(X)

        y_prob_0 = np.where(w == 1, y_predict_1[:, 0], y_predict_0[:, 0])

        y_prob_1 = np.where(w == 1, y_predict_1[:, 1], y_predict_0[:, 1])

        return np.column_stack((y_prob_0, y_prob_1))

    def predict_ite(self, X):
        check_is_fitted(self)
        X = self._scale_input_data_if_indicated(X)

        try:
            y_predict_0 = self.knn_control_model.predict_proba(X)[:, 1]
            y_predict_1 = self.knn_treated_model.predict_proba(X)[:, 1]
        except IndexError:
            print(
                "Positivity has been violated: either control or treatment group has only one y class."
            )
            raise

        return y_predict_1 - y_predict_0
