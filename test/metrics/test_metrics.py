import numpy as np
import pytest

from pycausal_explorer.metrics import mape, pehe

def test_mape_metric():
    x = np.array([1, 1, 1])
    y = np.array([1, 1, 4])
    result = mape(x, y)
    assert result == 1

def test_pehe_metric():
    x = np.array([1, 1, 1, 5])
    y = np.array([1, 1, 1, 1])
    result = pehe(x, y)
    assert result == 2