import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from pycausal_explorer.datasets.synthetic import create_synthetic_data
from pycausal_explorer.meta import RALearner


def test_ralearner_init_learner():
    learner = LinearRegression()
    ralearner = RALearner(LinearRegression())
    assert type(ralearner.u0) is type(learner)
    assert type(ralearner.u1) is type(learner)
    assert type(ralearner.tau) is type(learner)


def test_ralearner_init_custom_learner():
    drlearner = RALearner(
        learner=None,
        u0=LinearRegression(),
        u1=LinearRegression(),
        tau=RandomForestRegressor(n_estimators=100),
    )
    assert isinstance(drlearner.u0, LinearRegression)
    assert isinstance(drlearner.u1, LinearRegression)
    assert isinstance(drlearner.tau, RandomForestRegressor)


def test_ralearner_init_raise_exception():
    with pytest.raises(ValueError):
        RALearner(u0=LinearRegression(), u1=LinearRegression())


def test_ralearner_train():
    x, w, y = create_synthetic_data(random_seed=42)
    ralearner = RALearner(learner=LinearRegression())
    ralearner.fit(x, y, treatment=w)
    model_ate = ralearner.predict_ate(x)
    assert 1.0 == pytest.approx(model_ate, 0.0001)
