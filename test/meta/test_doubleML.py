import pytest

from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC

from pycausal_explorer.datasets.synthetic import create_synthetic_data

from pycausal_explorer.meta import DoubleMLBinaryTreatment, DoubleMLLinear


def test_doubleml_init():
    double_learner = DoubleMLLinear(LinearRegression(), LinearRegression())
    assert isinstance(double_learner, BaseEstimator)


def test_doubleml_train_binary():
    x, w, y = create_synthetic_data(random_seed=42)

    double_learner = DoubleMLBinaryTreatment(
        RandomForestRegressor(), SVC(probability=True)
    )

    double_learner.fit(x, y, treatment=w)
    _ = double_learner.predict_ite(x)
    model_ate = double_learner.predict_ate(x)
    assert 1.0 == pytest.approx(model_ate, 0.1)


def test_doubleml_train_linear():
    x, w, y = create_synthetic_data(random_seed=42)

    double_learner = DoubleMLLinear(RandomForestRegressor(), RandomForestRegressor())

    double_learner.fit(x, y, treatment=w)
    _ = double_learner.predict_ite(x)
    _ = double_learner.predict_ate(x)
    # Linear doubleML performs pretty badly on binary treatment datasets, so no tests on the ATE yet


def test_doubleml_train_linear_orthogonalscore():
    x, w, y = create_synthetic_data(random_seed=42)

    double_learner = DoubleMLLinear(
        RandomForestRegressor(), RandomForestRegressor(), score="orthogonal"
    )

    double_learner.fit(x, y, treatment=w)
    _ = double_learner.predict_ite(x)
    _ = double_learner.predict_ate(x)
    # Linear doubleML performs pretty badly on binary treatment datasets, so no tests on the ATE yet


def test_doubleml_invalide_score():
    x, w, y = create_synthetic_data(random_seed=42)

    with pytest.raises(ValueError):
        _ = DoubleMLLinear(
            RandomForestRegressor(), RandomForestRegressor(), score="invalid"
        )
