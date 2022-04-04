Model Guide
-------------
In this section, we outline each model strong and weak points and recommended
use cases.

Meta learners
==============
Meta learners rely on machine learning models that predict outcome.

They perform poorly when the outcome is hard to predict, such as when there
are too many covariables or the generating function is too complex. They are also
strongly affected by regularization.

Bellow we outline each one. Overall, unless the SLearner's bias towards zero
is desirable, the XLearner has the best convergence rate.

SLearner
""""""""""
The Slearner treats the treatment indicator like any
other predictor. [1] (https://arxiv.org/pdf/1706.03461.pdf ). It is often biased towards 0,
and is more accurate when treatment effect is often low.

TLearner
""""""""""
The Tlearner does not combine the treated and control groups, but instead
creates a model for each. It's less data efficient, but performs better when both
groups behave very differently.

XLearner
""""""""""
The XLearner applies a machine learning model to predict treatment effect, on top
of the ones to predict treatment and control groups. It performs better all around,
and particularly well when there are many more entries in one group than in the other.

Linear
""""""""""
Linear models are versions of the SLearner restricted to the LinearRegressor
and LogisticRegressor models. They are implemented mostly for educational purposes.


Causal Forests
=================
The causal forests model makes use of two different models: a random forest to reduce
noise, and a KNN to match the feature to the nearby features that went trough treatment,
and the ones that did not.

Causal forests are very good at predicting complex treatment effects, that vary strongly
and non-linearly between features. They also preform wel

Nearest Neighbor
""""""""""""""""""""
The nearest neighbors model aims to find similar data points to the one being analysed.
It generally performs worse than the causal forests model.


https://arxiv.org/pdf/1706.03461.pdf (meta learners)
https://www.statworx.com/en/content-hub/blog/machine-learning-goes-causal-ii-meet-the-random-forests-causal-brother/ (causal forests)

