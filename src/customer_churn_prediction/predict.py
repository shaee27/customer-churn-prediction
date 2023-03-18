from pathlib import Path

import click
import mlflow.sklearn
import pandas as pd


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/test.csv",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to test dataset",
)
@click.option(
    "--model-name",
    type=str,
    help="Model name in MLflow registry.",
)
@click.option(
    "--model-version",
    default="1",
    show_default=True,
    type=str,
    help="Model version in MLflow registry.",
)
def predict(
    dataset_path: Path,
    model_name: str,
    model_version: str,
) -> None:
    mlflow.set_tracking_uri("http://0.0.0.0:5000/")
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")
    X_test = pd.read_csv(dataset_path)
    submission = pd.read_csv("data/submission.csv")
    submission["Churn"] = model.predict_proba(X_test)[:, 1]
    submission.to_csv("my submission.csv", index=False)
