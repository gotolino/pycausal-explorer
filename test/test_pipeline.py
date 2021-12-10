from sklearn.preprocessing import StandardScaler

from causal_learn.datasets.synthetic import create_synthetic_data
from causal_learn.pipeline import Pipeline
from causal_learn.reweight import IPTW


def test_pipeline_fit():
    x, w, y = create_synthetic_data()
    pipe = Pipeline([("norm", StandardScaler()), ("clf", IPTW())])
    pipe.fit(x, y, clf__treatment=w)
    _ = pipe.predict_ite(x)
