import click
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import Dict, Type

from .data import FeaturePreprocessor
import customer_churn_prediction.model as mlflow_model


MODELS: Dict[str, Type[mlflow_model.MLflowModel]] = {
    "logreg": mlflow_model.LogisticRegressionMLflow,
    "rf": mlflow_model.RandomForestMLflow,
    "knn": mlflow_model.KnnMLflow,
    "catboost": mlflow_model.CatBoostMLflow,
    "lgbm": mlflow_model.LgbmMLflow,
    "tabnet": mlflow_model.TabNetMLflow,
    "stacking": mlflow_model.StackingMLflow,
}

NUM_COLS = [
    "ClientPeriod",
    "MonthlySpending",
    "TotalSpent",
]

CAT_COLS = [
    "Sex",
    "IsSeniorCitizen",
    "HasPartner",
    "HasChild",
    "HasPhoneService",
    "HasMultiplePhoneNumbers",
    "HasInternetService",
    "HasOnlineSecurityService",
    "HasOnlineBackup",
    "HasDeviceProtection",
    "HasTechSupportAccess",
    "HasOnlineTV",
    "HasMovieSubscription",
    "HasContractPhone",
    "IsBillingPaperless",
    "PaymentMethod",
]

FEATURE_COLS = NUM_COLS + CAT_COLS
TARGET_COL = "Churn"


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
    scale: bool,
    ohe: bool,
    model: str,
    run_name: str,
) -> None:
    dataset = pd.read_csv(dataset_path)
    click.echo(f"Dataset shape: {dataset.shape}.")
    features = dataset.drop(TARGET_COL, axis=1)
    target = dataset[TARGET_COL]
    if test_split_ratio > 0:
        (
            features_train,
            features_val,
            target_train,
            target_val,
        ) = train_test_split(
            features,
            target,
            test_size=test_split_ratio,
            random_state=random_state,
        )
    else:
        features_train = features
        target_train = target

    pipeline = Pipeline([("feature_preprocessor", FeaturePreprocessor())])
    if scale or ohe:
        if scale == "all" and ohe:
            raise ValueError(
                "Can't use one-hot encoding after scaling categorical features. "
                "If you want to use OHE, please scale only for numerical"
                "features (option `num`)or don't use at all (`none`)."
            )
        ct = ColumnTransformer([])
        if scale == "num":
            ct.transformers.append(("num", StandardScaler(), NUM_COLS))
        if scale == "all":
            ct.transformers.append(("num", StandardScaler(), FEATURE_COLS))
        if ohe:
            if model == "catboost":
                raise ValueError(
                    "Do not use one-hot encoding with CatBoost. This affects "
                    "both the training speed and the resulting quality."
                )
            ct.transformers.append(("cat", OneHotEncoder(), CAT_COLS))
        if ct.transformers:
            pipeline.steps.append(("num_cat_preprocessor", ct))

    class_name = MODELS[model].__name__
    click.echo(f"Training {class_name}")
    classifier = MODELS[model](
        pipeline=pipeline, random_state=random_state, cat_features=CAT_COLS
    )

    classifier.train_with_logging(features_train, target_train, run_name)
    if test_split_ratio > 0:
        roc_auc = classifier.evaluate(features_val, target_val)
        click.echo(f"ROC AUC score: {roc_auc}.")
