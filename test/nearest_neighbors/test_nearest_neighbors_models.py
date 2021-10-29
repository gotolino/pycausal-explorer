import numpy as np
import pandas as pd
import pytest

from causal_learn.datasets.synthetic import create_synthetic_data
from causal_learn.nearest_neighbors import CausalKNNClassifier, CausalKNNRegressor


def test_causal_knn_classifier_init():
    causal_knn_classifier = CausalKNNClassifier()
    assert type(causal_knn_classifier.params) is dict
    assert type(causal_knn_classifier.covariates) is list
    assert type(causal_knn_classifier.treatment) is str
    assert causal_knn_classifier._estimator_type == 'classifier'


def test_causal_knn_regressor_init():
    causal_knn_regressor = CausalKNNRegressor()
    assert type(causal_knn_regressor.params) is dict
    assert type(causal_knn_regressor.covariates) is list
    assert type(causal_knn_regressor.treatment) is str
    assert causal_knn_regressor._estimator_type == 'regressor'


def test_causal_knn_classifier_init_custom():
    causal_knn_classifier = CausalKNNClassifier(
        params={'n_neighbors': 20},
        covariates=['x0', 'x1'],
        treatment='t'
    )
    assert 'n_neighbors' in causal_knn_classifier.params.keys()
    assert causal_knn_classifier.params['n_neighbors'] == 20
    assert len(causal_knn_classifier.covariates) == 2
    assert causal_knn_classifier.covariates[0] == 'x0'
    assert causal_knn_classifier.covariates[1] == 'x1'
    assert causal_knn_classifier.treatment == 't'
    assert causal_knn_classifier._estimator_type == 'classifier'


def test_causal_knn_regressor_init_custom():
    causal_knn_regressor = CausalKNNRegressor(
        params={'n_neighbors': 20},
        covariates=['x0', 'x1'],
        treatment='t'
    )
    assert 'n_neighbors' in causal_knn_regressor.params.keys()
    assert causal_knn_regressor.params['n_neighbors'] == 20
    assert len(causal_knn_regressor.covariates) == 2
    assert causal_knn_regressor.covariates[0] == 'x0'
    assert causal_knn_regressor.covariates[1] == 'x1'
    assert causal_knn_regressor.treatment == 't'
    assert causal_knn_regressor._estimator_type == 'regressor'


def test_causal_knn_classifier_init_raise_exception():
    with pytest.raises(ValueError):
        CausalKNNClassifier(
            params=20
        )
        CausalKNNClassifier(
            covariates='x0'
        )
        CausalKNNClassifier(
            treatment=['t']
        )


def test_causal_knn_regressor_init_raise_exception():
    with pytest.raises(ValueError):
        CausalKNNRegressor(
            params=20
        )
        CausalKNNRegressor(
            covariates='x0'
        )
        CausalKNNRegressor(
            treatment=['t']
        )


def test_causal_knn_classifier_train():
    x, w, y = create_synthetic_data(random_seed=42, target_type='categorical')
    data = pd.DataFrame(columns=['x0', 't', 'y'], data=np.column_stack([x, w, y]))

    causal_knn_classifier = CausalKNNClassifier(
        params={'n_neighbors': 20},
        covariates=['x0'],
        treatment='t'
    )

    causal_knn_classifier.fit(X=data[['t', 'x0']], y=data['y'])
    _ = causal_knn_classifier.predict(X=data[['t', 'x0']])
    _ = causal_knn_classifier.predict_proba(X=data[['t', 'x0']])
    _ = causal_knn_classifier.predict_ite(X=data[['x0']])
    _ = causal_knn_classifier.predict_ate(X=data[['x0']])


def test_causal_knn_regressor_train():
    x, w, y = create_synthetic_data(random_seed=42)
    data = pd.DataFrame(columns=['x0', 't', 'y'], data=np.column_stack([x, w, y]))

    causal_knn_regressor = CausalKNNRegressor(
        params={'n_neighbors': 20},
        covariates=['x0'],
        treatment='t'
    )

    causal_knn_regressor.fit(X=data[['t', 'x0']], y=data['y'])
    _ = causal_knn_regressor.predict(X=data[['t', 'x0']])
    _ = causal_knn_regressor.predict_ite(X=data[['x0']])
    _ = causal_knn_regressor.predict_ate(X=data[['x0']])
