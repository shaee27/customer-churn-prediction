from click.testing import CliRunner
from pathlib import Path
import pytest
import os

from customer_churn_prediction.train import train
from customer_churn_prediction.data import TRAIN_DATA_PATH


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def generate_test_dataset(path: Path) -> None:
    os.makedirs(path.parent)
    open(path, "a").close()


def test_error_training_catboost_with_ohe(runner: CliRunner) -> None:
    """It fails when model is catboost and one-hot encoding is enabled."""
    with runner.isolated_filesystem():
        generate_test_dataset(Path(TRAIN_DATA_PATH))
        result = runner.invoke(train, ["--ohe", "--model", "catboost"])
        assert result.exit_code == 1
        msg = "Do not use one-hot encoding with CatBoost"
        assert msg in str(result.exception)


def test_error_for_unknown_model(runner: CliRunner) -> None:
    """It fails when model name is unknown."""
    result = runner.invoke(train, ["-m", "yetti"])
    assert result.exit_code == 2
    assert "Invalid value for '-m' / '--model'" in result.output


def test_error_for_invalid_random_state(runner: CliRunner) -> None:
    """It fails when passing invalid value for random_state option."""
    result = runner.invoke(train, ["--random-state", "-1"])
    assert result.exit_code == 2
    assert "Invalid value for '--random-state'" in result.output

    result = runner.invoke(train, ["--random-state", "4294967296"])
    assert result.exit_code == 2
    assert "Invalid value for '--random-state'" in result.output
