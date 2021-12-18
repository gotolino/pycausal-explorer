from sklearn.preprocessing import StandardScaler

from pycausal_explorer.datasets.synthetic import create_synthetic_data
from pycausal_explorer.pipeline import Pipeline
from pycausal_explorer.reweight import IPTW


def test_pipeline_fit():
    x, w, y = create_synthetic_data()
    pipe = Pipeline([("norm", StandardScaler()), ("clf", IPTW())])
    pipe.fit(x, y, clf__treatment=w)
    _ = pipe.predict_ite(x)
