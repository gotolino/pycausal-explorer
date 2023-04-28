import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from pycausal_explorer.datasets.synthetic import create_synthetic_data
from pycausal_explorer.meta import DRLearner


def test_drlearner_init_learner():
    learner = LinearRegression()
    drlearner = DRLearner(LinearRegression())
    assert type(drlearner.u0[0]) is type(learner)
    assert type(drlearner.u1[0]) is type(learner)
    assert type(drlearner.tau[0]) is type(learner)

    assert type(drlearner.u0[1]) is type(learner)
    assert type(drlearner.u1[1]) is type(learner)
    assert type(drlearner.tau[1]) is type(learner)


def test_drlearner_init_custom_learner():
    drlearner = DRLearner(
        learner=None,
        u0=LinearRegression(),
        u1=LinearRegression(),
        tau=RandomForestRegressor(n_estimators=100),
    )
    assert isinstance(drlearner.u0[0], LinearRegression)
    assert isinstance(drlearner.u1[0], LinearRegression)
    assert isinstance(drlearner.tau[0], RandomForestRegressor)

    assert isinstance(drlearner.u0[1], LinearRegression)
    assert isinstance(drlearner.u1[1], LinearRegression)
    assert isinstance(drlearner.tau[1], RandomForestRegressor)


def test_drlearner_init_raise_exception():
    with pytest.raises(ValueError):
        DRLearner(u0=LinearRegression(), u1=LinearRegression())


def test_drlearner_train():
    x, w, y = create_synthetic_data(random_seed=42)
    drlearner = DRLearner(learner=LinearRegression())
    drlearner.fit(x, y, treatment=w)
    model_ate = drlearner.predict_ate(x)
    assert 1.0 == pytest.approx(model_ate, 0.0001)
