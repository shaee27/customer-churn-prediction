import click
from pathlib import Path
from typing import Dict, Type

from .data import get_dataset, CAT_COLS, TRAIN_DATA_PATH
import customer_churn_prediction.model as mlflow_model
from .pipeline import create_pipeline


MODELS: Dict[str, Type[mlflow_model.MLflowModel]] = {
    "logreg": mlflow_model.LogisticRegressionMLflow,
    "rf": mlflow_model.RandomForestMLflow,
    "knn": mlflow_model.KnnMLflow,
    "catboost": mlflow_model.CatBoostMLflow,
    "lgbm": mlflow_model.LgbmMLflow,
    "tabnet": mlflow_model.TabNetMLflow,
    "stacking": mlflow_model.StackingMLflow,
}


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default=TRAIN_DATA_PATH,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-s",
    "--scale",
    default="num",
    show_default=True,
    type=click.Choice(["num", "all", "none"]),
    help=(
        "Use standard scaler: "
        "num - for numerical features, "
        "all - for all features, "
        "none - don't use scaler"
    ),
)
@click.option(
    "--ohe/--no-ohe",
    default=True,
    show_default=True,
    type=bool,
    help="Use one-hot encoding.",
)
@click.option(
    "-m",
    "--model",
    default="logreg",
    show_default=True,
    type=click.Choice(list(MODELS.keys())),
)
@click.option(
    "-r",
    "--run-name",
    default=None,
    type=str,
    help="MLflow run name.",
)
@click.option(
    "--random-state",
    default=0,
    show_default=True,
    type=click.IntRange(0, 2**32 - 1),
)
def train(
    dataset_path: Path,
    random_state: int,
    scale: str,
    ohe: bool,
    model: str,
    run_name: str,
) -> None:
    if ohe and model == "catboost":
        raise ValueError(
            "Do not use one-hot encoding with CatBoost. This affects both the "
            "training speed and the resulting quality."
        )

    features, target = get_dataset(dataset_path, random_state)

    class_name = MODELS[model].__name__
    click.echo(f"Training {class_name}")

    classifier = MODELS[model](
        pipeline=create_pipeline(scale, ohe),
        random_state=random_state,
        cat_features=CAT_COLS,
    )
    classifier.train_with_logging(features, target, run_name)
