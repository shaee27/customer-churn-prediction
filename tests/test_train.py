from click.testing import CliRunner
import pytest

from customer_churn_prediction.train import train


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_error_for_invalid_test_split_ratio(
    runner: CliRunner
) -> None:
    """It fails when test split ratio is greater than 1."""
    result = runner.invoke(train, ["--test-split-ratio", 2])
    assert result.exit_code == 2
    assert "Invalid value for '--test-split-ratio'" in result.output

