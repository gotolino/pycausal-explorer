import pytest

from pycausal_explorer.datasets.synthetic import create_synthetic_data
from pycausal_explorer.nearest_neighbors import PSM
from pycausal_explorer.reweight import PropensityScore


def test_propensity_score_matching_init():
    psm = PSM()
    assert isinstance(psm.propensity_score, PropensityScore)


def test_propensity_score_matching_train():
    x, w, y = create_synthetic_data(random_seed=42, target_type="binary")

    psm = PSM()

    psm.fit(x, y, treatment=w)
    _ = psm.predict(x, w)
    _ = psm.predict_ite(x)
    _ = psm.predict_ate(x)
