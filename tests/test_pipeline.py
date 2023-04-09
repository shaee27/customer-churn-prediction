import pytest
from sklearn.compose._column_transformer import ColumnTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from customer_churn_prediction.data import NUM_COLS, CAT_COLS
from customer_churn_prediction.pipeline import create_pipeline


def test_error_ohe_with_scale_all() -> None:
    """It fails when OHE is used after scaling all features."""
    with pytest.raises(ValueError) as e:
        create_pipeline(scale="all", ohe=True)
    assert "OHE cannot be used on scaled features." in str(e.value)


def test_scale_num_returns_correct_pipeline() -> None:
    """It checks that scale="num" returns correct pipeline."""
    pipeline = create_pipeline(scale="num", ohe=False)
    assert type(pipeline.steps[1][1]) == ColumnTransformer
    assert pipeline.steps[1][1].transformers[0][2] == NUM_COLS


def test_scale_all_returns_correct_pipeline() -> None:
    """It checks that scale="all" returns correct pipeline."""
    pipeline = create_pipeline(scale="all", ohe=False)
    assert type(pipeline.steps[1][1]) == ColumnTransformer
    assert pipeline.steps[1][1].transformers[0][2] == NUM_COLS + CAT_COLS


def test_scale_none_returns_correct_pipeline() -> None:
    """It checks that scale="all" returns correct pipeline."""
    pipeline = create_pipeline(scale="none", ohe=False)
    assert len(pipeline.steps) == 1


def test_ohe_returns_correct_pipeline() -> None:
    """It checks that ohe=True returns correct pipeline."""
    pipeline = create_pipeline(scale="none", ohe=True)
    assert type(pipeline.steps[1][1].transformers[0][1]) == OneHotEncoder
