import click
from mlflow import MlflowClient


@click.command()
@click.option(
    "-n",
    "--name",
    type=str,
    help="Model name in MLflow registry.",
)
@click.option(
    "-v",
    "--version",
    type=int,
    default=None,
    help="Model version.",
)
def delete_model(
    name: str,
    version: int,
) -> None:
    """Delete a model from MLflow registry."""
    if version is not None:
        MlflowClient().delete_model_version(name=name, version=version)
    else:
        MlflowClient().delete_registered_model(name=name)
