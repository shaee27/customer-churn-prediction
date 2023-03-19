import click
from pathlib import Path
from typing import Dict, Type

from .data import get_dataset, CAT_COLS
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
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--random-state", default=42, type=int)
@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, max_open=True),
    help="The proportion of the dataset to include in the test split.",
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
def train(
    dataset_path: Path,
    random_state: int,
    test_split_ratio: float,
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

    features_train, features_val, target_train, target_val = get_dataset(
        dataset_path,
        random_state,
        test_split_ratio,
    )

    class_name = MODELS[model].__name__
    click.echo(f"Training {class_name}")

    classifier = MODELS[model](
        pipeline=create_pipeline(scale, ohe),
        random_state=random_state,
        cat_features=CAT_COLS,
    )
    classifier.train_with_logging(features_train, target_train, run_name)

    if test_split_ratio > 0:
        roc_auc = classifier.evaluate(features_val, target_val)
        click.echo(f"ROC AUC score: {roc_auc}.")
