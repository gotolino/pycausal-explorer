[![codecov](https://codecov.io/gh/gotolino/causal-learn/branch/main/graph/badge.svg?token=5W6KVR73GJ)](https://codecov.io/gh/gotolino/causal-learn)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Template #

This README would normally document whatever steps are necessary to get your application up and running.

### What is this repository for? ###

* Quick summary
* Version
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

### How do I get set up? ###

* Summary of set up

If you want to just install and use the library, go to the project folder and install with `pip install .`

If you are going to develop the library, go to the project folder and install with `pip install -e .[dev]`.
It will install the development dependencies and the `-e` flag (editable) will allow us to edit the files and test it 
without needing to install again. You should also install the pre-commit hook `pre-commit install`. 
That will allow you to use the pre-commit setup, with isort, back formatting and flake8 style checker.

* Dependencies

All dependencies are described in the `setup.py` file, split into dependencies and dev dependencies.
* How to run tests

To run tests simply run `pytest` in your environment.

* Deployment instructions

### Contribution guidelines ###

* Writing tests

    Methods and classes should have proper tests using the pytest framework. 
All tests should pass before any merge being considerate.

* Code review

    Any pull-request should be reviewed by two other main contributors. 
Proper tests should be made and all automations should pass before considering merging.

* Other guidelines

    The project linting and style enforcing is done by [black](https://github.com/psf/black), [flake8](https://flake8.pycqa.org/en/latest/) and [isort](https://pypi.org/project/isort/).
There is a pre-commit hook that take care of checking those before the commit is made.

    Methods should have type hinting and a docstring following the [numpy](https://numpydoc.readthedocs.io/en/latest/format.html) convention.
[Pyment](https://pypi.org/project/pyment/) can help to automatically create proper docstrings.

### Who do I talk to? ###

* Repo owner or admin
* Other community or team contact
