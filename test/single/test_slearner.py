import numpy as np
import pandas as pd
import pytest

from pycausal_explorer.datasets.synthetic import create_synthetic_data
from pycausal_explorer.single import SingleLearner

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator


def test_causal_single_learner_init():
    single_learner = SingleLearner(LinearRegression())
    assert isinstance(single_learner.learner, BaseEstimator)


def test_causal_single_learner_train_linear():
    x, w, y = create_synthetic_data(random_seed=42)

    single_learner = SingleLearner(LinearRegression())

    single_learner.fit(x, y, treatment=w)
    _ = single_learner.predict(x, w)
    _ = single_learner.predict_ite(x)
    model_ate = single_learner.predict_ate(x)
    assert 1.0 == pytest.approx(model_ate, 0.0001)


def test_causal_single_learner_train_forest():
    x, w, y = create_synthetic_data(random_seed=42)

    single_learner = SingleLearner(RandomForestRegressor())

    single_learner.fit(x, y, treatment=w)
    _ = single_learner.predict(x, w)
    _ = single_learner.predict_ite(x)
    model_ate = single_learner.predict_ate(x)
    assert 1.0 == pytest.approx(model_ate, 0.0001)
