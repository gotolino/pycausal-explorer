import numpy as np
import pandas as pd
import pytest

from causal_learn.datasets.synthetic import create_synthetic_data
from causal_learn.reweight import IPTW, PropensityScore


def test_iptw_init():
    iptw = IPTW()
    assert isinstance(iptw.propensity_score, PropensityScore)


def test_propensity_score_fit():
    x, w, y = create_synthetic_data(random_seed=42)
    X_treatment = pd.DataFrame(np.column_stack([x, w]), columns=["x1", "treatment"])
    iptw = IPTW()
    iptw.fit(X_treatment, y)
    model_ite = iptw.predict_ite(x)
    model_ate = iptw.predict_ate(x)
    np.testing.assert_array_almost_equal(model_ite, np.ones(1000), decimal=2)
    assert 1.0 == pytest.approx(model_ate, 0.02)
