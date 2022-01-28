"""
    Data downloaded from the supplemental materials tab available at: https://www.tandfonline.com/doi/suppl/10.1198/jcgs.2010.08162?scroll=top

    The Infant Health and Development Program was a collaborative, randomized,
    longitudinal, multisite clinical trial designed to evaluate the efficacy of
    comprehensive early intervention in reducing the developmental and health
    problems of low birth weight, premature infants. An intensive intervention extending
    from hospital discharge to 36 months corrected age was administered
    between 1985 and 1988 at eight different sites. [GROSS, R. et al., 1993]

    Simulated outcomes are computed as described in [HILL, J., 2011]
    Starting with the experimental data, an observational study is created by
    throwing away a nonrandom portion of the treatment group: all children with nonwhite mothers.
    This leaves 139 children. The control group remains intact with 608 children. Thus the treatment
    and control groups are no longer balanced and simple comparisons of outcomes would lead to biased
    estimates of the treatment effect. Ethnicity was chosen as the variable
    used to partition the data because it led to subgroups that were more distinct than those yielded
    by the other categorical variables.

    The order of the variables from left to right is:
    - treat: treatment indicator (1 if treated, 0 if not treated);
    - bw: weight at birth (kg)
    - b.head: head circumference at birth (inches)
    - preterm: preterm indicator (1 if treated, 0 if not treated);
    - birth.o: birth order of baby by mother
    - nnhealth: neonatal health score
    - momage: mother's age at birth
    - sex
    - twin
    - b.marr: was mother married at birth
    - mom.lths
    - mom.hs: mother went to high school
    - mom.scoll: mother scholarity at birth
    - cig: mother smoked during pregnancy
    - first: mother's first baby
    - booze: mother drank alcohol during pregnancy
    - drugs: mother smoked during pregnancy
    - work.dur: mother worked during pregnancy
    - prenatal: mother went through prenatal treatment
    - ark, ein, har, mia, pen, tex, was: one-hot encoded columns indicating place of enrollment in the program
    - y_A_sim: simulated outcome for surface A (linear) as described on [HILL, J., 2011]
    - y_B_sim: simulated outcome for surface B (non-linear) as described on [HILL, J., 2011]
"""

from os import path
import pandas as pd

from ._filenames import DATASETS_PATH, IHDP_SIMULATED_OUTCOMES_CSV_FILE, ROOT_DIR


def load_ihdp_simulated_outcomes_dataset():
    return pd.read_csv(
        path.join(ROOT_DIR, DATASETS_PATH, IHDP_SIMULATED_OUTCOMES_CSV_FILE)
    )
