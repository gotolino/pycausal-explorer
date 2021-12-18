from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator


class BaseCausalModel(BaseEstimator, ABC):
    """ "Base class for causal model.

    All models should inherit from this base class.
    It should at least implement a fit and predict_ite methods.
    """

    @abstractmethod
    def fit(self, X, y, *, treatment):
        """
        Fit the model with variables X and target y.
        Parameters
        ----------
        X: array-like feature matrix
        y: array-like target
        treatment: array-like treatment column
        """

    @abstractmethod
    def predict_ite(self, X):
        """
        Predict the individual treatment effect for the model.

        If the model does not have an individual treatment effect,
        return the average treatment effect with the same shape of X.shape[0].
        Parameters
        ----------
        X: array-like feature matrix

        Returns
        -------
        treatment_effect : ndarray of results
        """

    def predict_ate(self, X):
        """
        Predict the average treatment effect for the model.

        Parameters
        ----------
        X: array-like feature matrix

        Returns
        -------
        treatment_effect : float result
        """
        return np.mean(self.predict_ite(X))
