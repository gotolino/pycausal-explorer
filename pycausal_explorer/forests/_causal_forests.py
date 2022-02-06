import numpy as np
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.utils.validation import check_is_fitted, check_X_y

from pycausal_explorer.base import BaseCausalModel

from ._constants import (
    forest_classifier_algorithms_dict,
    forest_regressor_algorithms_dict,
    supported_forest_algorithms,
)


class BaseCausalForest(BaseCausalModel):
    def __init__(
        self,
        forest_algorithm="extratrees",
        knn_params=None,
        random_search_params=None,
        model_search_params=None,
    ):

        if (
            type(forest_algorithm) is not str
            or forest_algorithm not in supported_forest_algorithms
        ):
            raise ValueError(
                "Algorithm name must be a string among the options: 'extratrees', 'random_forest', 'xgboost'"
            )

        if knn_params and type(knn_params) is not dict:
            raise ValueError("KNN params must be a dictionary")

        if random_search_params and type(random_search_params) is not dict:
            raise ValueError("Random Search params must be a dictionary")

        if model_search_params and type(model_search_params) is not dict:
            raise ValueError("Model Search params must be a dictionary")

        self.forest_algorithm = forest_algorithm

        self.knn_params = knn_params
        if not knn_params:
            self.knn_params = {
                "n_neighbors": 10,
                "metric": "hamming",
            }

        self.random_search_params = random_search_params
        if not random_search_params:
            self.random_search_params = {
                "n_iter": 65,
                "cv": 3,
                "scoring": "neg_mean_absolute_percentage_error",
                "n_jobs": 10,
                "random_state": 1,
            }

        self.model_search_params = model_search_params
        if not model_search_params:
            self.model_search_params = {
                "n_estimators": randint(10, 500),
                "max_depth": randint(3, 20),
                "max_features": ["auto", "sqrt", "log2"],
            }


class CausalForestRegressor(BaseCausalForest):
    """
    Implementation of the Causal forests model.

    It makes use of decision trees and K nearest neighbors models to find similar data points, and compares
    their outcome when under treatment and when under control to find the effect of treatment.

    Parameters
    ----------
    forest_algorithm : basestring
        Which forest algorithm to use. One of "extratrees", random_forest" or"xgboost".

    knn_params : dict
        Parameters to train KNeighborsRegressor from sklearn.neighbors
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html

    random_search_params : dict
        Randomized Search Parameters to be uses by RandomizedSearchCV from sklearn.model_selection
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html

    model_search_params=None : dict
        Model Search Parameters to be uses by RandomizedSearchCV from sklearn.model_selection
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
    """

    def __init__(
        self,
        forest_algorithm="extratrees",
        knn_params=None,
        random_search_params=None,
        model_search_params=None,
    ):
        super().__init__(
            forest_algorithm=forest_algorithm,
            knn_params=knn_params,
            random_search_params=random_search_params,
            model_search_params=model_search_params,
        )
        self._estimator_type = "regressor"

    def fit(self, X, y, *, treatment):
        X, y = check_X_y(X, y)
        X, w = check_X_y(X, treatment)

        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
            X, y, w, test_size=0.5, random_state=42
        )

        random_search_model = forest_regressor_algorithms_dict[self.forest_algorithm](
            random_state=42
        )
        random_search = RandomizedSearchCV(
            random_search_model, self.model_search_params, **self.random_search_params
        )

        random_search_results = random_search.fit(X_train, y_train)

        self.fitted_model = random_search_results.best_estimator_
        self.fitted_model_params_ = random_search_results.best_params_

        self.feature_importances_ = (
            random_search_results.best_estimator_.feature_importances_
        )

        leaves_val = self.fitted_model.apply(X_val)

        # Train KNN model for the control group with the validation set
        self.knn_control = KNeighborsRegressor(**self.knn_params).fit(
            X=leaves_val[w_val == 0, :],
            y=y_val[w_val == 0],
        )

        # Train KNN model for the treated with the validation set
        self.knn_treated = KNeighborsRegressor(**self.knn_params).fit(
            X=leaves_val[w_val == 1, :],
            y=y_val[w_val == 1],
        )
        self.is_fitted_ = True
        return self

    def predict(self, X, w):
        check_is_fitted(self)
        leaves = self.fitted_model.apply(X)

        # Predict y0 for the test set
        y_predict_0 = self.knn_control.predict(X=leaves)

        # Predict y1 for the test set
        y_predict_1 = self.knn_treated.predict(X=leaves)

        return np.where(w == 1, y_predict_1, y_predict_0)

    def predict_ite(self, X):
        check_is_fitted(self)
        leaves = self.fitted_model.apply(X)

        # Predict y0 for the test set
        y_predict_0 = self.knn_control.predict(X=leaves)

        # Predict y1 for the test set
        y_predict_1 = self.knn_treated.predict(X=leaves)

        return y_predict_1 - y_predict_0


class CausalForestClassifier(BaseCausalForest):
    """
    Implementation of the Causal forests model.

    It makes use of decision trees and K nearest neighbors models to find similar data points, and compares
    their outcome when under treatment and when under control to find the effect of treatment.

    Parameters
    ----------
    forest_algorithm : basestring
        Which forest algorithm to use. One of "extratrees", random_forest" or"xgboost".

    knn_params : dict
        Parameters to train KNeighborsRegressor from sklearn.neighbors
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html

    random_search_params : dict
        Randomized Search Parameters to be uses by RandomizedSearchCV from sklearn.model_selection
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html

    model_search_params=None : dict
        Model Search Parameters to be uses by RandomizedSearchCV from sklearn.model_selection
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
    """

    def __init__(
        self,
        forest_algorithm="extratrees",
        knn_params=None,
        random_search_params=None,
        model_search_params=None,
    ):
        super().__init__(
            forest_algorithm=forest_algorithm,
            knn_params=knn_params,
            random_search_params=random_search_params,
            model_search_params=model_search_params,
        )
        self._estimator_type = "classifier"

    def fit(self, X, y, *, treatment):
        X, y = check_X_y(X, y)
        X, w = check_X_y(X, treatment)

        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
            X, y, w, test_size=0.5, random_state=42
        )

        random_search_model = forest_classifier_algorithms_dict[self.forest_algorithm](
            random_state=42
        )
        random_search = RandomizedSearchCV(
            random_search_model, self.model_search_params, **self.random_search_params
        )

        random_search_results = random_search.fit(X=X_train, y=y_train)

        self.fitted_model = random_search_results.best_estimator_
        self.fitted_model_params_ = random_search_results.best_params_

        self.feature_importances_ = (
            random_search_results.best_estimator_.feature_importances_
        )

        leaves_val = self.fitted_model.apply(X_val)

        # Train KNN model for the control group with the validation set
        self.knn_control = KNeighborsClassifier(**self.knn_params).fit(
            X=leaves_val[w_val == 0, :],
            y=y_val[w_val == 0],
        )

        # Train KNN model for the treated with the validation set
        self.knn_treated = KNeighborsClassifier(**self.knn_params).fit(
            X=leaves_val[w_val == 1, :],
            y=y_val[w_val == 1],
        )
        self.is_fitted_ = True
        return self

    def predict(self, X, w):
        check_is_fitted(self)
        leaves = self.fitted_model.apply(X)

        # Predict y0 for the test set
        y_predict_0 = self.knn_control.predict(X=leaves)

        # Predict y1 for the test set
        y_predict_1 = self.knn_treated.predict(X=leaves)

        return np.where(w == 1, y_predict_1, y_predict_0)

    def predict_proba(self, X, w):
        check_is_fitted(self)
        leaves = self.fitted_model.apply(X)

        # Predict y0 for the test set
        y_predict_0 = self.knn_control.predict_proba(X=leaves)

        # Predict y1 for the test set
        y_predict_1 = self.knn_treated.predict_proba(X=leaves)

        y_prob_0 = np.where(w == 1, y_predict_1[:, 0], y_predict_0[:, 0])

        y_prob_1 = np.where(w == 1, y_predict_1[:, 1], y_predict_0[:, 1])

        return np.column_stack((y_prob_0, y_prob_1))

    def predict_ite(self, X):
        check_is_fitted(self)
        leaves = self.fitted_model.apply(X)

        try:
            # Predict y0 for the test set
            y_predict_0 = self.knn_control.predict_proba(X=leaves)[:, 1]

            # Predict y1 for the test set
            y_predict_1 = self.knn_treated.predict_proba(X=leaves)[:, 1]
        except IndexError:
            print(
                "Positivity has been violated: either control or treatment group has only one y class."
            )
            raise

        return y_predict_1 - y_predict_0
