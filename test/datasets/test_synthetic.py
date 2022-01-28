import pytest

from pycausal_explorer.datasets.synthetic import create_synthetic_data


def test_synthetic_creation():
    size = 1000
    x, w, y = create_synthetic_data(size)
    assert x.shape[0] == 1000
    assert w.shape[0] == 1000
    assert y.shape[0] == 1000


def test_synthetic_creation_treatment():
    size = 1000
    x, w, y = create_synthetic_data(size)
    assert w.sum() / size > 0.4


def test_synthetic_data_creation_random_seed():
    size = 1000
    x, w, y = create_synthetic_data(size, random_seed=42)
    assert x[0] == pytest.approx(1.4967141530112327)
    assert w[0] == 1
    assert y[0] == pytest.approx(1.7483570765056164)
