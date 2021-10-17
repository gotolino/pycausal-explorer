import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from causal_learn.datasets.synthetic import create_synthetic_data
from causal_learn.meta import XLearner


def test_xlearner_init_learner():
    learner = LinearRegression()
    xlearner = XLearner(LinearRegression())
    assert type(xlearner.u0) is type(learner)
    assert type(xlearner.u1) is type(learner)
    assert type(xlearner.te_u0) is type(learner)
    assert type(xlearner.te_u1) is type(learner)


def test_xlearner_init_custom_learner():
    learner1 = LinearRegression()
    learner2 = RandomForestRegressor(n_estimators=100)
    xlearner = XLearner(
        u0=LinearRegression(),
        u1=LinearRegression(),
        te_u0=RandomForestRegressor(n_estimators=100),
        te_u1=RandomForestRegressor(n_estimators=100),
    )
    assert type(xlearner.u0) is type(learner1)
    assert type(xlearner.u1) is type(learner1)
    assert type(xlearner.te_u0) is type(learner2)
    assert type(xlearner.te_u1) is type(learner2)


def test_xlearner_init_raise_exception():
    with pytest.raises(ValueError):
        XLearner(u0=LinearRegression(), u1=LinearRegression())


def test_xlearner_train():
    x, w, y = create_synthetic_data(random_seed=42)
    xlearner = XLearner(learner=LinearRegression())
    xlearner.fit(x, w, y)
    _ = xlearner.predict(x, w)
    _ = xlearner.predict_ate(x)
