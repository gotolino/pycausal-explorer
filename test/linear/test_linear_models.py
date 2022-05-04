import numpy as np
import pandas as pd
import pytest

from pycausal_explorer.datasets.synthetic import create_synthetic_data
from pycausal_explorer.linear import CausalLinearRegression, CausalLogisticRegression


def test_causal_logistic_regression_init():
    causal_logistic_regression = CausalLogisticRegression()
    assert causal_logistic_regression._estimator_type == "classifier"


def test_causal_linear_regression_init():
    causal_linear_regression = CausalLinearRegression()
    assert causal_linear_regression._estimator_type == "regressor"


def test_causal_logistic_regression_train():
    x, w, y = create_synthetic_data(random_seed=42, target_type="binary")

    causal_logistic_regression = CausalLogisticRegression()

    causal_logistic_regression.fit(x, y, treatment=w)
    _ = causal_logistic_regression.predict(x, w)
    _ = causal_logistic_regression.predict(
        np.concatenate((x, w.reshape((-1, 1))), axis=1)
    )
    _ = causal_logistic_regression.predict_proba(x, w)
    _ = causal_logistic_regression.predict_ite(x)


def test_causal_linear_regression_train():
    x, w, y = create_synthetic_data(random_seed=42)

    causal_linear_regression = CausalLinearRegression()

    causal_linear_regression.fit(x, y, treatment=w)
    _ = causal_linear_regression.predict(x, w)
    _ = causal_linear_regression.predict(
        np.concatenate((x, w.reshape((-1, 1))), axis=1)
    )
    _ = causal_linear_regression.predict_ite(x)
    model_ate = causal_linear_regression.predict_ate(x)
    assert 1.0 == pytest.approx(model_ate, 0.0001)
