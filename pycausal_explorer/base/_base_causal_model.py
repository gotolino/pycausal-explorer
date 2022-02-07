from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator


class BaseCausalModel(BaseEstimator, ABC):
    """Base class for causal model.

    All models should inherit from this base class.
    All models should at least implement a fit and predict_ite methods.
    """

    @abstractmethod
    def fit(self, X, y, *, treatment):
        """
        Fit model with variables X and target y.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Features to control for when estimating causal effect.

        y : array-like of shape (n_samples,)
            Outcome of samples.

        treatment : array-like of shape (n_samples,)
            Binary array. Describes wether or not treatment was applied on a given sample.

        Returns
        -------
        self : object
            Fitted model.
        """

    @abstractmethod
    def predict_ite(self, X):
        """
        Predict the individual treatment effect for the model.

        If the model does not have an individual treatment effect,
        return the average treatment effect with the same shape of X.shape[0].
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Features of each sample

        Returns
        -------
        treatment_effect : ndarray of results
        """

    def predict_ate(self, X):
        """
        Predict the average treatment effect for the model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Features of each sample

        Returns
        -------
        treatment_effect : float result
        """
        return np.mean(self.predict_ite(X))
