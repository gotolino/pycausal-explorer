from pycausal_explorer.datasets.lalonde_nsw_jobs import load_lalonde_nsw_jobs_dataset


CSV_COLUMNS = [
    "data_id",
    "treat",
    "age",
    "education",
    "married",
    "nodegree",
    "black",
    "hispanic",
    "white",
    "re75",
    "re78",
]


def test_load_lalonde_nsw_jobs_dataset():
    df = load_lalonde_nsw_jobs_dataset()
    assert len(df) > 0
    assert len(df.columns) == len(CSV_COLUMNS)
    for column in CSV_COLUMNS:
        assert column in df.columns
