import numpy as np
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from causal_learn.base import BaseCausalModel
from ._constants import forest_classifier_algorithms_dict, forest_regressor_algorithms_dict, supported_forest_algorithms

class CausalForestRegressor(BaseCausalModel):
    def __init__(self, forest_algorithm='extratrees', covariates=[], treatment='', knn_params={}):
        self._estimator_type = 'regressor'

        if type(forest_algorithm) is not str or forest_algorithm not in supported_forest_algorithms:
            raise ValueError(
                "Algorithm name must be a string among the options: 'extratrees', 'random_forest', 'xgboost'"
            )

        if type(covariates) is not list:
            raise ValueError("Covariates must be a list")
        if type(treatment) is not str:
            raise ValueError("Treatment must be a string")
        if type(knn_params) is not dict:
            raise ValueError("KNN params must be a dictionary")

        self.forest_algorithm = forest_algorithm
        self.covariates = covariates
        self.treatment = treatment

        self.knn_params = knn_params
        if not knn_params:
            self.knn_params = {
                'n_neighbors': 10,
                'metric': 'hamming',
            }

    def fit(self, X, y):
        model_params = {
            'n_estimators': randint(10, 500),
            'max_depth': randint(3, 20),
            'max_features': ['auto', 'sqrt', 'log2']
        }

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.5,
            random_state=42
        )

        random_search_model = forest_regressor_algorithms_dict[self.forest_algorithm](random_state=42)
        random_search = RandomizedSearchCV(
            random_search_model,
            model_params,
            n_iter=65,
            cv=3,
            scoring='neg_mean_absolute_percentage_error',
            n_jobs=10,
            random_state=1
        )

        random_search_results = random_search.fit(
            X=X_train[self.covariates],
            y=y_train
        )

        self.fitted_model = random_search_results.best_estimator_
        self.fitted_model_params_ = random_search_results.best_params_

        self.feature_importances_ = random_search_results.best_estimator_.feature_importances_

        leaves_val = self.fitted_model.apply(X_val[self.covariates])

        # Train KNN model for the control group with the validation set
        self.knn_control = KNeighborsRegressor(
            **self.knn_params
        ).fit(
            X=leaves_val[X_val.reset_index().query(f'{self.treatment} == 0').index.tolist(), :],
            y=y_val.loc[X_val.query(f'{self.treatment} == 0').index]
        )

        # Train KNN model for the treated with the validation set
        self.knn_treated = KNeighborsRegressor(
            **self.knn_params
        ).fit(
            X=leaves_val[X_val.reset_index().query(f'{self.treatment} == 1').index.tolist(), :],
            y=y_val.loc[X_val.query(f'{self.treatment} == 1').index]
        )

    def predict(self, X):
        leaves = self.fitted_model.apply(X[self.covariates])

        # Predict y0 for the test set
        y_predict_0 = self.knn_control.predict(X=leaves)

        # Predict y1 for the test set
        y_predict_1 = self.knn_treated.predict(X=leaves)

        return np.where(
            X[self.treatment].values == 1,
            y_predict_1,
            y_predict_0
        )

    def predict_ite(self, X):
        leaves = self.fitted_model.apply(X[self.covariates])

        # Predict y0 for the test set
        y_predict_0 = self.knn_control.predict(X=leaves)

        # Predict y1 for the test set
        y_predict_1 = self.knn_treated.predict(X=leaves)

        return y_predict_1 - y_predict_0


class CausalForestClassifier(BaseCausalModel):
    def __init__(self, forest_algorithm='extratrees', covariates=[], treatment='', knn_params={}):
        self._estimator_type = 'classifier'

        if type(forest_algorithm) is not str or forest_algorithm not in supported_forest_algorithms:
            raise ValueError(
                "Algorithm name must be a string among the options: 'extratrees', 'random_forest', 'xgboost'"
            )
        if type(covariates) is not list:
            raise ValueError("Covariates must be a list")
        if type(treatment) is not str:
            raise ValueError("Treatment must be a string")
        if type(knn_params) is not dict:
            raise ValueError("KNN params must be a dictionary")

        self.forest_algorithm = forest_algorithm
        self.covariates = covariates
        self.treatment = treatment

        self.knn_params = knn_params
        if not knn_params:
            self.knn_params = {
                'n_neighbors': 10,
                'metric': 'hamming',
            }

    def fit(self, X, y):
        model_params = {
            'n_estimators': randint(10, 500),
            'max_depth': randint(3, 20),
            'max_features': ['auto', 'sqrt', 'log2']
        }

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.5,
            random_state=42
        )

        random_search_model = forest_classifier_algorithms_dict[self.forest_algorithm](random_state=42)
        random_search = RandomizedSearchCV(
            random_search_model,
            model_params,
            n_iter=65,
            cv=3,
            scoring='neg_mean_absolute_percentage_error',
            n_jobs=10,
            random_state=1
        )

        random_search_results = random_search.fit(
            X=X_train[self.covariates],
            y=y_train
        )

        self.fitted_model = random_search_results.best_estimator_
        self.fitted_model_params_ = random_search_results.best_params_

        self.feature_importances_ = random_search_results.best_estimator_.feature_importances_

        leaves_val = self.fitted_model.apply(X_val[self.covariates])

        # Train KNN model for the control group with the validation set
        self.knn_control = KNeighborsClassifier(
            **self.knn_params
        ).fit(
            X=leaves_val[X_val.reset_index().query(f'{self.treatment} == 0').index.tolist(), :],
            y=y_val.loc[X_val.query(f'{self.treatment} == 0').index]
        )

        # Train KNN model for the treated with the validation set
        self.knn_treated = KNeighborsClassifier(
            **self.knn_params
        ).fit(
            X=leaves_val[X_val.reset_index().query(f'{self.treatment} == 1').index.tolist(), :],
            y=y_val.loc[X_val.query(f'{self.treatment} == 1').index]
        )

    def predict(self, X):
        leaves = self.fitted_model.apply(X[self.covariates])

        # Predict y0 for the test set
        y_predict_0 = self.knn_control.predict(X=leaves)

        # Predict y1 for the test set
        y_predict_1 = self.knn_treated.predict(X=leaves)

        return np.where(
            X[self.treatment].values == 1,
            y_predict_1,
            y_predict_0
        )

    def predict_proba(self, X):
        leaves = self.fitted_model.apply(X[self.covariates])

        # Predict y0 for the test set
        y_predict_0 = self.knn_control.predict_proba(X=leaves)

        # Predict y1 for the test set
        y_predict_1 = self.knn_treated.predict_proba(X=leaves)

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
        leaves = self.fitted_model.apply(X[self.covariates])

        try:
            # Predict y0 for the test set
            y_predict_0 = self.knn_control.predict_proba(X=leaves)[:, 1]

            # Predict y1 for the test set
            y_predict_1 = self.knn_treated.predict_proba(X=leaves)[:, 1]
        except IndexError:
            print('Positivity has been violated: either control or treatment group has only one y class.')

        return y_predict_1 - y_predict_0

