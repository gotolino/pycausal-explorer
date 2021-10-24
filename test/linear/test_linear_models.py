import numpy as np
import pandas as pd
import pytest

from causal_learn.datasets.synthetic import create_synthetic_data
from causal_learn.linear import CausalLinearRegression, CausalLogisticRegression


def test_causal_logistic_regression_init():
    causal_logistic_regression = CausalLogisticRegression()
    assert type(causal_logistic_regression.covariates) is list
    assert type(causal_logistic_regression.treatment) is str
    assert causal_logistic_regression._estimator_type == 'classifier'


def test_causal_linear_regression_init():
    causal_linear_regression = CausalLinearRegression()
    assert type(causal_linear_regression.covariates) is list
    assert type(causal_linear_regression.treatment) is str
    assert causal_linear_regression._estimator_type == 'regressor'


def test_causal_logistic_regression_init_custom():
    causal_logistic_regression = CausalLogisticRegression(
        covariates=['x0', 'x1'],
        treatment='t'
    )
    assert len(causal_logistic_regression.covariates) == 2
    assert causal_logistic_regression.covariates[0] == 'x0'
    assert causal_logistic_regression.covariates[1] == 'x1'
    assert causal_logistic_regression.treatment == 't'
    assert causal_logistic_regression._estimator_type == 'classifier'


def test_causal_linear_regression_init_custom():
    causal_linear_regression = CausalLinearRegression(
        covariates=['x0', 'x1'],
        treatment='t'
    )
    assert len(causal_linear_regression.covariates) == 2
    assert causal_linear_regression.covariates[0] == 'x0'
    assert causal_linear_regression.covariates[1] == 'x1'
    assert causal_linear_regression.treatment == 't'
    assert causal_linear_regression._estimator_type == 'regressor'


def test_causal_logistic_regression_init_raise_exception():
    with pytest.raises(ValueError):
        CausalLogisticRegression(
            covariates='x0'
        )
        CausalLogisticRegression(
            treatment=['t']
        )


def test_causal_linear_regression_init_raise_exception():
    with pytest.raises(ValueError):
        CausalLinearRegression(
            covariates='x0'
        )
        CausalLinearRegression(
            treatment=['t']
        )


def test_causal_logistic_regression_train():
    x, w, y = create_synthetic_data(random_seed=42, target_type='categorical')
    data = pd.DataFrame(columns=['x0', 't', 'y'], data=np.column_stack([x, w, y]))

    causal_logistic_regression = CausalLogisticRegression(
        covariates=['x0'],
        treatment='t'
    )

    causal_logistic_regression.fit(X=data[['t', 'x0']], y=data['y'])
    _ = causal_logistic_regression.predict(X=data[['t', 'x0']])
    _ = causal_logistic_regression.predict_proba(X=data[['t', 'x0']])
    _ = causal_logistic_regression.predict_ite(X=data[['x0']])


def test_causal_linear_regression_train():
    x, w, y = create_synthetic_data(random_seed=42)
    data = pd.DataFrame(columns=['x0', 't', 'y'], data=np.column_stack([x, w, y]))

    causal_linear_regression = CausalLinearRegression(
        covariates=['x0'],
        treatment='t'
    )

    causal_linear_regression.fit(X=data[['t', 'x0']], y=data['y'])
    _ = causal_linear_regression.predict(X=data[['t', 'x0']])
    _ = causal_linear_regression.predict_ite(X=data[['x0']])
