from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import FeaturePreprocessor, CAT_COLS, NUM_COLS


def create_pipeline(scale: str, ohe: bool) -> Pipeline:
    if ohe and scale == "all":
        raise ValueError(
            "OHE cannot be used on scaled features. Scale only numerical "
            "features (num) or use no scaling (none) to apply OHE."
        )

    steps = [("feature preprocessor", FeaturePreprocessor())]
    transformers = []
    if scale == "num":
        transformers.append(("scale", StandardScaler(), NUM_COLS))
    if scale == "all":
        transformers.append(("scale", StandardScaler(), NUM_COLS + CAT_COLS))
    if ohe:
        transformers.append(("ohe", OneHotEncoder(), CAT_COLS))
    if transformers:
        steps.append(("scale_ohe", ColumnTransformer(transformers)))

    return Pipeline(steps)
