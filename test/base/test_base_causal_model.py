import numpy as np
import pytest

from pycausal_explorer.base import BaseCausalModel


class ModelWithoutFit(BaseCausalModel):
    def predict_ite(self, X):
        pass


class ModelWithoutPredictIte(BaseCausalModel):
    def fit(self, X, y):
        pass


class MyModel(BaseCausalModel):
    def fit(self, X, y):
        pass

    def predict_ite(self, X):
        return np.ones(X.shape[0])


def test_model_inheritance():
    model = MyModel()
    assert isinstance(model, BaseCausalModel)


def test_model_without_fit():
    with pytest.raises(TypeError):
        _ = ModelWithoutFit()


def test_model_without_predict_ite():
    with pytest.raises(TypeError):
        _ = ModelWithoutPredictIte()


def test_model_predict():
    model = MyModel()
    X = np.ones([100, 10])
    model_ite = model.predict_ite(X)
    model_ate = model.predict_ate(X)

    np.testing.assert_array_equal(model_ite, np.ones(100))
    assert model_ate == 1.0
