import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from pycausal_explorer.datasets.synthetic import create_synthetic_data
from pycausal_explorer.meta import XLearner


def test_xlearner_init_learner():
    learner = LinearRegression()
    xlearner = XLearner(LinearRegression())
    assert type(xlearner.u0) is type(learner)
    assert type(xlearner.u1) is type(learner)
    assert type(xlearner.te_u0) is type(learner)
    assert type(xlearner.te_u1) is type(learner)


def test_xlearner_init_custom_learner():
    xlearner = XLearner(
        learner=None,
        u0=LinearRegression(),
        u1=LinearRegression(),
        te_u0=RandomForestRegressor(n_estimators=100),
        te_u1=RandomForestRegressor(n_estimators=100),
    )
    assert isinstance(xlearner.u0, LinearRegression)
    assert isinstance(xlearner.u1, LinearRegression)
    assert isinstance(xlearner.te_u0, RandomForestRegressor)
    assert isinstance(xlearner.te_u1, RandomForestRegressor)


def test_xlearner_init_raise_exception():
    with pytest.raises(ValueError):
        XLearner(u0=LinearRegression(), u1=LinearRegression())


def test_xlearner_train():
    x, w, y = create_synthetic_data(random_seed=42)
    xlearner = XLearner(learner=LinearRegression())
    xlearner.fit(x, y, treatment=w)
    _ = xlearner.predict(x, w)
    _ = xlearner.predict_ite(x)
