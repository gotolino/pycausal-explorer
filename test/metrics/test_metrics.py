import pytest

from pycausal_explorer.metrics import mape, pehe

def test_mape_metric():
    x = [1, 1, 2]
    y = [1, 1, 1]
    result = mape(x, y)
    assert result == 1

def test_pehe_metric():
    x = [1, 1, 2]
    y = [1, 1, 5]
    result = pehe(x, y)
    assert result == 3