import numpy as np
import pandas as pd
import pytest

from causal_learn.datasets.synthetic import create_synthetic_data
from causal_learn.meta import CausalExtraTreesClassifier, CausalExtraTreesRegressor


def test_causal_extratrees_classifier_init():
    causal_extratrees_classifier = CausalExtraTreesClassifier()
    assert type(causal_extratrees_classifier.covariates) is list
    assert type(causal_extratrees_classifier.treatment) is str
    assert type(causal_extratrees_classifier.knn_params) is dict
    assert causal_extratrees_classifier._estimator_type == 'classifier'


def test_causal_extratrees_regressor_init():
    causal_extratrees_regressor = CausalExtraTreesRegressor()
    assert type(causal_extratrees_regressor.covariates) is list
    assert type(causal_extratrees_regressor.treatment) is str
    assert type(causal_extratrees_regressor.knn_params) is dict
    assert causal_extratrees_regressor._estimator_type == 'regressor'


def test_causal_extratrees_classifier_init_custom():
    causal_extratrees_classifier = CausalExtraTreesClassifier(
        covariates=['x0', 'x1'], treatment='t', knn_params={'n_neighbors': 20}
    )
    assert len(causal_extratrees_classifier.covariates) == 2
    assert causal_extratrees_classifier.covariates[0] == 'x0'
    assert causal_extratrees_classifier.covariates[1] == 'x1'
    assert causal_extratrees_classifier.treatment == 't'
    assert 'n_neighbors' in causal_extratrees_classifier.knn_params.keys()
    assert causal_extratrees_classifier.knn_params['n_neighbors'] == 20
    assert causal_extratrees_classifier._estimator_type == 'classifier'


def test_causal_extratrees_regressor_init_custom():
    causal_extratrees_regressor = CausalExtraTreesRegressor(
        covariates=['x0', 'x1'], treatment='t', knn_params={'n_neighbors': 20}
    )
    assert len(causal_extratrees_regressor.covariates) == 2
    assert causal_extratrees_regressor.covariates[0] == 'x0'
    assert causal_extratrees_regressor.covariates[1] == 'x1'
    assert causal_extratrees_regressor.treatment == 't'
    assert 'n_neighbors' in causal_extratrees_regressor.knn_params.keys()
    assert causal_extratrees_regressor.knn_params['n_neighbors'] == 20
    assert causal_extratrees_regressor._estimator_type == 'regressor'


def test_causal_extratrees_classifier_init_raise_exception():
    with pytest.raises(ValueError):
        CausalExtraTreesClassifier(
            covariates='x0', treatment=['t'], knn_params=20
        )


def test_causal_extratrees_regressor_init_raise_exception():
    with pytest.raises(ValueError):
        CausalExtraTreesRegressor(
            covariates='x0', treatment=['t'], knn_params=20
        )


# def test_causal_extratrees_classifier_train():
#     x, w, y = create_synthetic_data(random_seed=42)
#     y_class = np.where(y > 0, 1, 0)
#     data = pd.DataFrame(columns=['x0', 't', 'y'], data=np.column_stack([x, w, y_class]))
#
#     causal_extratrees_classifier = CausalExtraTreesClassifier(
#         covariates=['x0'],
#         treatment='t'
#     )
#
#     causal_extratrees_classifier.fit(X=data[['t', 'x0']], y=data['y'])
#     _ = causal_extratrees_classifier.predict(X=data[['t', 'x0']])
#     _ = causal_extratrees_classifier.predict_ate(X=data[['x0']])
#
#
# def test_causal_extratrees_regressor_train():
#     x, w, y = create_synthetic_data(random_seed=42)
#     data = pd.DataFrame(columns=['x0', 't', 'y'], data=np.column_stack([x, w, y]))
#
#     causal_extratrees_regressor = CausalExtraTreesRegressor(
#         covariates=['x0'],
#         treatment='t'
#     )
#
#     causal_extratrees_regressor.fit(X=data[['t', 'x0']], y=data['y'])
#     _ = causal_extratrees_regressor.predict(X=data[['t', 'x0']])
#     _ = causal_extratrees_regressor.predict_ate(X=data[['x0']])
