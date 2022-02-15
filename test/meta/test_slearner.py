import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

from pycausal_explorer.datasets.synthetic import create_synthetic_data
from pycausal_explorer.meta import SingleLearnerRegressor


def test_causal_single_learner_init():
    single_learner = SingleLearnerRegressor(LinearRegression())
    assert isinstance(single_learner.learner, BaseEstimator)


def test_causal_single_learner_train_linear():
    x, w, y = create_synthetic_data(random_seed=42)

    single_learner = SingleLearnerRegressor(LinearRegression())

    single_learner.fit(x, y, treatment=w)
    _ = single_learner.predict(x, w)
    _ = single_learner.predict_ite(x)
    model_ate = single_learner.predict_ate(x)
    assert 1.0 == pytest.approx(model_ate, 0.0001)


def test_causal_single_learner_train_polynomial():
    x, w, y = create_synthetic_data(random_seed=42)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(x.reshape(-1, 1))

    single_learner = SingleLearnerRegressor(LinearRegression())
    single_learner.fit(poly_features, y, treatment=w)

    model_ate = single_learner.predict_ate(poly_features)
    assert 1.0 == pytest.approx(model_ate, 0.0001)


def test_causal_single_learner_train_logistic():
    x, w, y = create_synthetic_data(random_seed=42)

    single_learner = SingleLearnerRegressor(LogisticRegression())
    single_learner.fit(x, y, treatment=w)
    model_ate = single_learner.predict_ate(x)

    assert model_ate > 0.5


def test_causal_single_learner_error():
    with pytest.raises(ValueError):
        _ = SingleLearnerRegressor(LinearRegression)
