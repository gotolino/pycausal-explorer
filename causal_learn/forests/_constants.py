from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from xgboost import XGBClassifier, XGBRegressor

supported_forest_algorithms = ["extratrees", "random_forest", "xgboost"]

forest_classifier_algorithms_dict = {
    "extratrees": ExtraTreesClassifier,
    "random_forest": RandomForestClassifier,
    "xgboost": XGBClassifier,
}

forest_regressor_algorithms_dict = {
    "extratrees": ExtraTreesRegressor,
    "random_forest": RandomForestRegressor,
    "xgboost": XGBRegressor,
}
