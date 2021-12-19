import pytest

from pycausal_explorer.datasets.synthetic import create_synthetic_data
from pycausal_explorer.nearest_neighbors import CausalKNNClassifier, CausalKNNRegressor


def test_causal_knn_classifier_init():
    causal_knn_classifier = CausalKNNClassifier()
    assert type(causal_knn_classifier.params) is dict
    assert causal_knn_classifier._estimator_type == "classifier"


def test_causal_knn_regressor_init():
    causal_knn_regressor = CausalKNNRegressor()
    assert type(causal_knn_regressor.params) is dict
    assert causal_knn_regressor._estimator_type == "regressor"


def test_causal_knn_classifier_init_custom():
    causal_knn_classifier = CausalKNNClassifier(params={"n_neighbors": 20})
    assert "n_neighbors" in causal_knn_classifier.params.keys()
    assert causal_knn_classifier.params["n_neighbors"] == 20
    assert causal_knn_classifier._estimator_type == "classifier"


def test_causal_knn_regressor_init_custom():
    causal_knn_regressor = CausalKNNRegressor(params={"n_neighbors": 20})
    assert "n_neighbors" in causal_knn_regressor.params.keys()
    assert causal_knn_regressor.params["n_neighbors"] == 20
    assert causal_knn_regressor._estimator_type == "regressor"


def test_causal_knn_classifier_init_raise_exception():
    with pytest.raises(ValueError):
        CausalKNNClassifier(params=20)


def test_causal_knn_regressor_init_raise_exception():
    with pytest.raises(ValueError):
        CausalKNNRegressor(params=20)


def test_causal_knn_classifier_train():
    x, w, y = create_synthetic_data(random_seed=42, target_type="categorical")

    causal_knn_classifier = CausalKNNClassifier(params={"n_neighbors": 20})

    causal_knn_classifier.fit(x, y, treatment=w)
    _ = causal_knn_classifier.predict(x, w)
    _ = causal_knn_classifier.predict_proba(x, w)
    _ = causal_knn_classifier.predict_ite(x)
    _ = causal_knn_classifier.predict_ate(x)


def test_causal_knn_regressor_train():
    x, w, y = create_synthetic_data(random_seed=42)

    causal_knn_regressor = CausalKNNRegressor(params={"n_neighbors": 20})

    causal_knn_regressor.fit(x, y, treatment=w)
    _ = causal_knn_regressor.predict(x, w)
    _ = causal_knn_regressor.predict_ite(x)
    model_ate = causal_knn_regressor.predict_ate(x)
    assert 1.0 == pytest.approx(model_ate, 0.02)
