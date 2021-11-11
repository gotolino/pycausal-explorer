[![codecov](https://codecov.io/gh/gotolino/causal-learn/branch/main/graph/badge.svg?token=5W6KVR73GJ)](https://codecov.io/gh/gotolino/causal-learn)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Causal Learn #

Causal Learn is a python module for causal inference and treatment effect estimation. It implements a set of 
algorithms that supports causal analysis.

## Installation Guide ##

You can install the package through pip:

``pip install causal_learn``

## Basic Usage ##

```
from causal_learn.datasets.synthetic import create_syntetic_dataset
from causal_learn.meta impot XLearner

x, treatment, y = create_synthetic_dataset()
model = XLearner()
model.fit(x, treatment, y)
treatment_effect = model.predict_ite(x)
```