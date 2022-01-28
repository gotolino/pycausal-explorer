"""
    Data sets were downloaded from NYU website: http://users.nber.org/~rdehejia/nswdata2.html

    The data are drawn from a paper by Robert Lalonde,
    "Evaluating the Econometric Evaluations of Training Programs,"
    American Economic Review, Vol. 76, pp. 604-620.
    We are grateful to him for allowing us to use this data,
    assistance in reading his original data tapes, and permission to publish it here.

    The order of the variables from left to right is:
    - data_id:
        - "Lalonde Sample" if the sample belongs to the initial Lalonde controlled random trial
        - "PSID" if the sample belongs to the observational data collected after the controlled experiment
    - treat: treatment indicator (1 if treated, 0 if not treated);
    - age;
    - education: years of formal education;
    - married: 1 if married, 0 otherwise;
    - black: 1 if black, 0 otherwise;
    - hispanic: 1 if hispanic, 0 otherwise;
    - white: 1 if white, 0 otherwise;
    - nodegree: 1 if no degree, 0 otherwise;
    - re75: earnings in 1975;
    - re78: earnings in 1978.
"""

from os import path
import pandas as pd

from ._filenames import DATASETS_PATH, LALONDE_NSW_JOBS_CSV_FILE, ROOT_DIR


def load_lalonde_nsw_jobs_dataset():
    return pd.read_csv(path.join(ROOT_DIR, DATASETS_PATH, LALONDE_NSW_JOBS_CSV_FILE))
