import pytest
from scipy.stats import randint

from pycausal_explorer.datasets.synthetic import create_synthetic_data
from pycausal_explorer.forests import CausalForestClassifier, CausalForestRegressor


def test_causal_forest_classifier_init():
    causal_forest_classifier = CausalForestClassifier()
    assert type(causal_forest_classifier.forest_algorithm) is str
    assert type(causal_forest_classifier.knn_params) is dict
    assert type(causal_forest_classifier.random_search_params) is dict
    assert type(causal_forest_classifier.model_search_params) is dict
    assert causal_forest_classifier._estimator_type == "classifier"


def test_causal_forest_regressor_init():
    causal_forest_regressor = CausalForestRegressor()
    assert type(causal_forest_regressor.forest_algorithm) is str
    assert type(causal_forest_regressor.knn_params) is dict
    assert type(causal_forest_regressor.random_search_params) is dict
    assert type(causal_forest_regressor.model_search_params) is dict
    assert causal_forest_regressor._estimator_type == "regressor"


def test_causal_forest_classifier_init_custom():
    causal_forest_classifier = CausalForestClassifier(
        forest_algorithm="random_forest",
        knn_params={"n_neighbors": 20},
        random_search_params={"n_iter": 5, "cv": 3},
        model_search_params={
            "n_estimators": randint(10, 50),
            "max_depth": randint(3, 10),
        },
    )
    assert causal_forest_classifier.forest_algorithm == "random_forest"
    assert "n_neighbors" in causal_forest_classifier.knn_params.keys()
    assert causal_forest_classifier.knn_params["n_neighbors"] == 20
    assert "n_iter" in causal_forest_classifier.random_search_params.keys()
    assert causal_forest_classifier.random_search_params["n_iter"] == 5
    assert "n_estimators" in causal_forest_classifier.model_search_params.keys()
    assert causal_forest_classifier._estimator_type == "classifier"


def test_causal_forest_regressor_init_custom():
    causal_forest_regressor = CausalForestRegressor(
        forest_algorithm="random_forest",
        knn_params={"n_neighbors": 20},
        random_search_params={"n_iter": 5, "cv": 3},
        model_search_params={
            "n_estimators": randint(10, 50),
            "max_depth": randint(3, 10),
        },
    )
    assert causal_forest_regressor.forest_algorithm == "random_forest"
    assert "n_neighbors" in causal_forest_regressor.knn_params.keys()
    assert causal_forest_regressor.knn_params["n_neighbors"] == 20
    assert "n_iter" in causal_forest_regressor.random_search_params.keys()
    assert causal_forest_regressor.random_search_params["n_iter"] == 5
    assert "n_estimators" in causal_forest_regressor.model_search_params.keys()
    assert causal_forest_regressor._estimator_type == "regressor"


def test_causal_forest_classifier_init_raise_exception():
    with pytest.raises(ValueError):
        CausalForestClassifier(forest_algorithm="linear_regression")
        CausalForestClassifier(knn_params=20)


def test_causal_forest_regressor_init_raise_exception():
    with pytest.raises(ValueError):
        CausalForestRegressor(forest_algorithm="linear_regression")
        CausalForestRegressor(knn_params=20)


def test_causal_forest_classifier_train():
    x, w, y = create_synthetic_data(random_seed=42, target_type="binary")

    causal_forest_classifier = CausalForestClassifier(
        random_search_params={
            "n_iter": 5,
            "cv": 3,
            "n_jobs": 10,
            "random_state": 42,
        }
    )

    causal_forest_classifier.fit(x, y, treatment=w)
    _ = causal_forest_classifier.predict(x, w)
    _ = causal_forest_classifier.predict_proba(x, w)
    _ = causal_forest_classifier.predict_ite(x)


def test_causal_forest_regressor_train():
    x, w, y = create_synthetic_data(random_seed=42)

    causal_forest_regressor = CausalForestRegressor(
        random_search_params={
            "n_iter": 5,
            "cv": 3,
            "n_jobs": 10,
            "random_state": 42,
        }
    )

    causal_forest_regressor.fit(x, y, treatment=w)
    _ = causal_forest_regressor.predict(x, w)
    _ = causal_forest_regressor.predict_ite(x)
    model_ate = causal_forest_regressor.predict_ate(x)
    assert 1.0 == pytest.approx(model_ate, 0.02)
