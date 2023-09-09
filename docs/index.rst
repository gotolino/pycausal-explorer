.. pycausal explorer documentation master file, created by
   sphinx-quickstart on Wed Mar 16 14:11:31 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pycausal explorer's documentation!
=============================================



Hello, and welcome to our docs! pycausal explorer is a causal inference library mostly focused on
implementing models that predict causal effect.
If you don't know what causal inference is, you can check the `definition <https://en.wikipedia.org/wiki/Causal_inference>`_
or study it more comprehensively `using brandy neal's materials <https://www.bradyneal.com/causal-inference-course>`_.

Below we outline the libraries models and provide a :doc:`guide` on which to pick.
If you just want to see how it works, check out our :doc:`example`.

Install
-------------

To install pycausal explorer, simply run

.. code-block:: sh

    $ pip install pycausal_explorer

Models
-------------
Most of the features this library offers are models that predict causal effect. If you want to decide which model to use,
check out out :doc:`guide`. If you want to know every model we have, we also have a :doc:`model_list`.


* **Linear Learners**

Linear learners are simple models, that work best on linearly generated data.

  :mod:`pycausal_explorer.linear._linear_models`

|

* **Meta learners**

Meta learners make use of other machine learning models to predict causal effect.
Their effectiveness depends on how well the provided model can predict the relevant variables.

   | :mod:`pycausal_explorer.meta._single_learner`
   | :mod:`pycausal_explorer.meta._tlearner`
   | :mod:`pycausal_explorer.meta._xlearner`
   | :mod:`pycausal_explorer.meta._doubleML`

|

* **Causal Forests**

The Causal Forests model uses random forests as well as K Nearest Neighbors to estimate
treatment effect on complex, non-linear situations.
   :mod:`pycausal_explorer.forests._causal_forests`

|

* **Nearest Neighbors**

The Nearest Neighbors model will find the most similar element in the control and treatment groups, and use their difference
to find out the effect of treatment.
  :mod:`pycausal_explorer.nearest_neighbors._k_nearest_neighbors`

|

* **Propensity score**

The propensity score model aims to represent all covariates as a single scalar, the propensity for treatment to occur.
This simplifies the estimation of treatment effect.
  :mod:`pycausal_explorer.reweight._iptw`


Datasets
-------------

Pycausal explorer also offers datasets to validate causal inference models. Check them out :doc:`here<datasets>`



Indices
---------------------

* :ref:`genindex`
* :ref:`modindex`

.. toctree::
   :maxdepth: 2
   :caption: Models:

   model_list
   guide
   example

.. toctree::
   :maxdepth: 2
   :caption: Datasets:

   datasets