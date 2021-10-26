from causal_learn.datasets.synthetic import create_synthetic_data
from causal_learn.reweight import IPTW, PropensityScore


def test_iptw_init():
    iptw = IPTW()
    assert isinstance(iptw.propensity_score, PropensityScore)


def test_propensity_score_fit():
    x, w, y = create_synthetic_data(random_seed=42)
    iptw = IPTW()
    iptw.fit(x, w, y)
    _ = iptw.predict_ate(x)
