from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from pycausal_explorer.datasets.synthetic import create_synthetic_data
from pycausal_explorer.reweight import PropensityScore


def test_propensity_score_init():
    propensity_score = PropensityScore()
    assert isinstance(propensity_score.model, LogisticRegression)
    assert isinstance(propensity_score.scaler, StandardScaler)


def test_propensity_score_fit():
    x, w, _ = create_synthetic_data(random_seed=42)
    propensity_score = PropensityScore()
    propensity_score.fit(x, w)
    _ = propensity_score.predict(x)
    _ = propensity_score.predict_proba(x)
