import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from causal_learn.base import BaseCausalModel


class CausalKNNBaseModel(BaseCausalModel):
    def __init__(self, params={'n_neighbors': 10, 'metric': 'euclidean'}, covariates=[], treatment='', scale=True):
        if type(params) is not dict:
            raise ValueError("KNN params must be a dictionary")
        if type(covariates) is not list:
            raise ValueError("Covariates must be a list")
        if type(treatment) is not str:
            raise ValueError("Treatment must be a string")
        if type(scale) is not bool:
            raise ValueError("Scale flag must be a Boolean value")

        self.params = params
        self.covariates = covariates
        self.treatment = treatment
        self.scale = scale

        self.standard_scaler = StandardScaler()

    def _scale_input_data_if_indicated(self, X):
        if self.scale:
            X_array = self.standard_scaler.fit_transform(X[self.covariates])
            X[self.covariates] = X_array
        return X

    def fit(self, X, y):
        X = self._scale_input_data_if_indicated(X)

        # Train KNN model for the control group
        self.knn_control_model.fit(
            X=X.query(f'{self.treatment} == 0')[self.covariates],
            y=y.loc[X.query(f'{self.treatment} == 0').index]
        )

        # Train KNN model for the treated samples
        self.knn_treated_model.fit(
            X=X.query(f'{self.treatment} == 1')[self.covariates],
            y=y.loc[X.query(f'{self.treatment} == 1').index]
        )

    def predict(self, X):
        X = self._scale_input_data_if_indicated(X)

        # Predict y0 for the test set
        y_predict_0 = self.knn_control_model.predict(X[self.covariates])

        # Predict y1 for the test set
        y_predict_1 = self.knn_treated_model.predict(X[self.covariates])

        return np.where(
            X[self.treatment].values == 1,
            y_predict_1,
            y_predict_0
        )

    def predict_ite(self, X):
        X = self._scale_input_data_if_indicated(X)

        try:
            # Predict y0 for the test set
            y_predict_0 = self.knn_control_model.predict(X[self.covariates])

            # Predict y1 for the test set
            y_predict_1 = self.knn_treated_model.predict(X[self.covariates])
        except IndexError:
            print('Positivity has been violated: either control or treatment group has only one y class.')

        return y_predict_1 - y_predict_0


class CausalKNNRegressor(CausalKNNBaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._estimator_type = 'regressor'
        self.knn_control_model = KNeighborsRegressor(
            **self.params
        )
        self.knn_treated_model = KNeighborsRegressor(
            **self.params
        )


class CausalKNNClassifier(CausalKNNBaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._estimator_type = 'classifier'
        self.knn_control_model = KNeighborsClassifier(
            **self.params
        )
        self.knn_treated_model = KNeighborsClassifier(
            **self.params
        )

    def predict_proba(self, X):
        self._scale_input_data_if_indicated(X)

        # Predict y0 for the test set
        y_predict_0 = self.knn_control_model.predict_proba(X[self.covariates])

        # Predict y1 for the test set
        y_predict_1 = self.knn_treated_model.predict_proba(X[self.covariates])

        y_prob_0 = np.where(
            X[self.treatment].values == 1,
            y_predict_1[:, 0],
            y_predict_0[:, 0]
        )

        y_prob_1 = np.where(
            X[self.treatment].values == 1,
            y_predict_1[:, 1],
            y_predict_0[:, 1]
        )

        return np.column_stack((y_prob_0, y_prob_1))

    def predict_ite(self, X):
        X = self._scale_input_data_if_indicated(X)

        try:
            # Predict y0 for the test set
            y_predict_0 = self.knn_control_model.predict_proba(X[self.covariates])[:, 1]

            # Predict y1 for the test set
            y_predict_1 = self.knn_treated_model.predict_proba(X[self.covariates])[:, 1]
        except IndexError:
            print('Positivity has been violated: either control or treatment group has only one y class.')

        return y_predict_1 - y_predict_0
