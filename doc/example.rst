Example
-------------

.. TODO: Make dataset link to explanation about libraries datasets

This page goes over a simple usage of the library. We will train a model over a synthetic dataset(TODO: link here) and predict
the causal effect of the treatment variable.
Since the dataset is synthetic, we know the real causal effect and can validate the model.


.. TODO: Split code in manageable chunks and explain them

.. code-block:: python

    from pycausal_explorer.datasets.synthetic import create_synthetic_data
    from pycausal_explorer.reweight import IPTW

    x, w, y = create_synthetic_data(random_seed=42)
    linear_model = CausalLinearRegression()

    linear_model.fit(x, y, treatment=w)
    model_ate = linear_model.predict_ate(x)

    print(model_ate)
