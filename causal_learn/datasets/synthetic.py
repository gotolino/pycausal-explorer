import numpy as np


def create_synthetic_data(size=1000, target_type='continuous', random_seed=None):
    if random_seed:
        np.random.seed(random_seed)
    treatment_effect = 1
    x = np.random.normal(1, 1, size)
    chance_of_treatment = np.clip(0.5 + x / 10, 0, 1)
    treatment = np.random.binomial(1, chance_of_treatment, size)
    y = 0.5 * x + treatment_effect * treatment
    if target_type == 'categorical':
        y = np.where(y >= np.median(y), 1, 0)
    return x.reshape(-1, 1), treatment, y
