import numpy as np


def create_synthetic_data(size=1000, target_type="continuous", random_seed=None):
    """
    Creates a synthetic dataset with explicit causal effects.


    Parameters
    ----------
    size : int
        Amount of rows of created data.
    target_type : basestring
        "continuous" or "binary".
        Wether the outcome should be continuous or binary.
    random_seed : int, optional
        Random seed for data generation

    Returns
    -------
    out : ndarray
        Returns a 3 element tuple containing a common cause covariate,
        the treatment and the outcome.
    """
    if target_type not in ["continuous", "binary"]:
        raise ValueError(
            f"target_type must be 'continuous' or 'binary', {target_type} given"
        )

    if random_seed:
        np.random.seed(random_seed)
    treatment_effect = 1
    x = np.random.normal(1, 1, size)
    chance_of_treatment = np.clip(0.5 + x / 10, 0, 1)
    treatment = np.random.binomial(1, chance_of_treatment, size)
    y = 0.5 * x + treatment_effect * treatment
    if target_type == "binary":
        y = np.where(y >= np.median(y), 1, 0)
    return x.reshape(-1, 1), treatment, y
