import numpy as np
import pandas as pd
import pytest

from pycausal_explorer.datasets.synthetic import create_synthetic_data
from pycausal_explorer.forests import CausalForestClassifier, CausalForestRegressor


def test_causal_forest_classifier_init():
    causal_forest_classifier = CausalForestClassifier()
    assert type(causal_forest_classifier.forest_algorithm) is str
    assert type(causal_forest_classifier.covariates) is list
    assert type(causal_forest_classifier.treatment) is str
    assert type(causal_forest_classifier.knn_params) is dict
    assert causal_forest_classifier._estimator_type == "classifier"


def test_causal_forest_regressor_init():
    causal_forest_regressor = CausalForestRegressor()
    assert type(causal_forest_regressor.forest_algorithm) is str
    assert type(causal_forest_regressor.covariates) is list
    assert type(causal_forest_regressor.treatment) is str
    assert type(causal_forest_regressor.knn_params) is dict
    assert causal_forest_regressor._estimator_type == "regressor"


def test_causal_forest_classifier_init_custom():
    causal_forest_classifier = CausalForestClassifier(
        forest_algorithm="random_forest",
        covariates=["x0", "x1"],
        treatment="t",
        knn_params={"n_neighbors": 20},
    )
    assert causal_forest_classifier.forest_algorithm == "random_forest"
    assert len(causal_forest_classifier.covariates) == 2
    assert causal_forest_classifier.covariates[0] == "x0"
    assert causal_forest_classifier.covariates[1] == "x1"
    assert causal_forest_classifier.treatment == "t"
    assert "n_neighbors" in causal_forest_classifier.knn_params.keys()
    assert causal_forest_classifier.knn_params["n_neighbors"] == 20
    assert causal_forest_classifier._estimator_type == "classifier"


def test_causal_forest_regressor_init_custom():
    causal_forest_regressor = CausalForestRegressor(
        forest_algorithm="random_forest",
        covariates=["x0", "x1"],
        treatment="t",
        knn_params={"n_neighbors": 20},
    )
    assert causal_forest_regressor.forest_algorithm == "random_forest"
    assert len(causal_forest_regressor.covariates) == 2
    assert causal_forest_regressor.covariates[0] == "x0"
    assert causal_forest_regressor.covariates[1] == "x1"
    assert causal_forest_regressor.treatment == "t"
    assert "n_neighbors" in causal_forest_regressor.knn_params.keys()
    assert causal_forest_regressor.knn_params["n_neighbors"] == 20
    assert causal_forest_regressor._estimator_type == "regressor"


def test_causal_forest_classifier_init_raise_exception():
    with pytest.raises(ValueError):
        CausalForestClassifier(forest_algorithm="linear_regression")
        CausalForestClassifier(covariates="x0")
        CausalForestClassifier(treatment=["t"])
        CausalForestClassifier(knn_params=20)


def test_causal_forest_regressor_init_raise_exception():
    with pytest.raises(ValueError):
        CausalForestRegressor(forest_algorithm="linear_regression")
        CausalForestRegressor(covariates="x0")
        CausalForestRegressor(treatment=["t"])
        CausalForestRegressor(knn_params=20)


def test_causal_forest_classifier_train():
    x, w, y = create_synthetic_data(random_seed=42, target_type="categorical")
    data = pd.DataFrame(columns=["x0", "t", "y"], data=np.column_stack([x, w, y]))

    causal_forest_classifier = CausalForestClassifier(covariates=["x0"], treatment="t")

    causal_forest_classifier.fit(X=data[["t", "x0"]], y=data["y"])
    _ = causal_forest_classifier.predict(X=data[["t", "x0"]])
    _ = causal_forest_classifier.predict_proba(X=data[["t", "x0"]])
    _ = causal_forest_classifier.predict_ite(X=data[["x0"]])


def test_causal_forest_regressor_train():
    x, w, y = create_synthetic_data(random_seed=42)
    data = pd.DataFrame(columns=["x0", "t", "y"], data=np.column_stack([x, w, y]))

    causal_forest_regressor = CausalForestRegressor(covariates=["x0"], treatment="t")

    causal_forest_regressor.fit(X=data[["t", "x0"]], y=data["y"])
    _ = causal_forest_regressor.predict(X=data[["t", "x0"]])
    _ = causal_forest_regressor.predict_ite(X=data[["x0"]])
