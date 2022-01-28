[![codecov](https://codecov.io/gh/gotolino/pycausal-explorer/branch/main/graph/badge.svg?token=5W6KVR73GJ)](https://codecov.io/gh/gotolino/pycausal-explorer)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Pycausal-explorer #

Pycausal-explorer is a python module for causal inference and treatment effect estimation. It implements a set of 
algorithms that supports causal analysis.

## Installation Guide ##

You can install the package through pip:

``pip install pycausal-explorer``

## Basic Usage ##
All models are inherited from BaseCausalModel, that inherits from scikit-learn BaseEstimator. 
It uses scikit-learn framework to fit and predict the outcome. It implements predict_ite and predict_ate
methods that return the individual treatment effect and the average treatment effect, respectively.
```
from pycausal_explorer.datasets.synthetic import create_synthetic_data
from pycausal_explorer.meta import XLearner

x, treatment, y = create_synthetic_data()
model = XLearner()
model.fit(x, treatment, y)
treatment_effect = model.predict_ite(x)
```

## Current Implemented Models ##
This version currently implements propensity score and iptw in the reweight package, linear regression in the linear package, causal forests in forest package and x-learn in meta package. 

## Using Pipelines ##

Pycausal-explorer has a Pipeline class inherited from scikit-learn Pipeline. 
It implements the method predict_ite, so it can be used pro predict treatment effect in a pipeline:
```
from sklearn.preprocessing import StandardScaler

from pycausal_explorer.datasets.synthetic import create_synthetic_data
from pycausal_explorer.pipeline import Pipeline
from pycausal_explorer.reweight import IPTW

x, w, y = create_synthetic_data()
pipe = Pipeline([("norm", StandardScaler()), ("clf", IPTW())])
pipe.fit(x, y, clf__treatment=w)
treatment_effect = pipe.predict_ite(x)
```
