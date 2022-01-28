from pycausal_explorer.datasets.ihdp_simulated_outcomes import (
    load_ihdp_simulated_outcomes_dataset,
)


CSV_COLUMNS = columns_to_save = [
    "treat",
    "bw",
    "b.head",
    "preterm",
    "birth.o",
    "nnhealth",
    "momage",
    "sex",
    "twin",
    "b.marr",
    "mom.lths",
    "mom.hs",
    "mom.scoll",
    "cig",
    "first",
    "booze",
    "drugs",
    "work.dur",
    "prenatal",
    "ark",
    "ein",
    "har",
    "mia",
    "pen",
    "tex",
    "was",
    "y_A_sim",
    "y_B_sim",
]


def test_load_ihdp_simulated_outcomes_dataset():
    df = load_ihdp_simulated_outcomes_dataset()
    assert len(df) > 0
    assert len(df.columns) == len(CSV_COLUMNS)
    for column in CSV_COLUMNS:
        assert column in df.columns
