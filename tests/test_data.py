from click.testing import CliRunner
import os
import pandas as pd
from pathlib import Path
import pytest

from customer_churn_prediction.data import (
    get_dataset,
    FeaturePreprocessor,
    CAT_COLS,
    NUM_COLS,
    TARGET_COL,
    TRAIN_DATA_PATH,
)


TEST_DATA = [[
    0, 56.05, " ", "Female", 0, "Yes", "Yes", "No", "No phone service",
    "DSL", "Yes", "Yes", "Yes", "Yes", "Yes", "No", "Two year", "No",
    "Credit card (automatic)", 0
]]


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def generate_test_dataset(path: Path) -> None:
    os.makedirs(path.parent)
    pd.DataFrame(
        data=TEST_DATA,
        columns=NUM_COLS + CAT_COLS + [TARGET_COL],
    ).to_csv(path, index=False)


def test_get_dataset_returns_correct_dataset(runner: CliRunner) -> None:
    """Check dataset is loaded correctly."""
    with runner.isolated_filesystem():
        path = Path(TRAIN_DATA_PATH)
        generate_test_dataset(path)
        features, target = get_dataset(path, 0)
        assert features.shape == (1, 19)
        assert all(features.columns == NUM_COLS + CAT_COLS)
        assert target.shape == (1,)
        assert target.name == TARGET_COL


def test_feature_preprocessor_does_not_modify_original_data(
    runner: CliRunner,
) -> None:
    """Check original dataframe is not modified bby FeaturePreprocessor."""
    with runner.isolated_filesystem():
        path = Path(TRAIN_DATA_PATH)
        generate_test_dataset(path)
        features, target = get_dataset(path, 0)
        features_copy = features.copy()
        FeaturePreprocessor().fit_transform(features)
        assert features.equals(features_copy)


def test_feature_preprocessor_transforms_data_correctly(
    runner: CliRunner,
) -> None:
    """Check original dataframe is not modified bby FeaturePreprocessor."""
    with runner.isolated_filesystem():
        path = Path(TRAIN_DATA_PATH)
        generate_test_dataset(path)
        features, target = get_dataset(path, 0)
        t = FeaturePreprocessor().fit_transform(features).values.tolist()
        assert t == [[
            0.00, 56.05, 0.00, 1.00, 0.00, 1.00, 1.00, 0.00, 0.00, 1.00, 1.00,
            1.00, 1.00, 1.00, 1.00, 0.00, 2.00, 0.00, 0.00
        ]]
