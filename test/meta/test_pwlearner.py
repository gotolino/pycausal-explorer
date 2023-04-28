import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from pycausal_explorer.datasets.synthetic import create_synthetic_data
from pycausal_explorer.meta import PWLearner


def test_pwlearner_init_learner():
    learner = LinearRegression()
    pwlearner = PWLearner(LinearRegression())
    assert type(pwlearner.u0) is type(learner)
    assert type(pwlearner.u1) is type(learner)
    assert type(pwlearner.tau) is type(learner)


def test_pwlearner_init_custom_learner():
    pwlearner = PWLearner(
        learner=None,
        u0=LinearRegression(),
        u1=LinearRegression(),
        tau=RandomForestRegressor(n_estimators=100),
    )
    assert isinstance(pwlearner.u0, LinearRegression)
    assert isinstance(pwlearner.u1, LinearRegression)
    assert isinstance(pwlearner.tau, RandomForestRegressor)


def test_pwlearner_init_raise_exception():
    with pytest.raises(ValueError):
        PWLearner(u0=LinearRegression(), u1=LinearRegression())


def test_pwlearner_train():
    x, w, y = create_synthetic_data(random_seed=42)
    pwlearner = PWLearner(learner=LinearRegression())
    pwlearner.fit(x, y, treatment=w)
    model_ate = pwlearner.predict_ate(x)
    assert 1.0 == pytest.approx(model_ate, 0.1)
