from pathlib import Path

import click
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import FeaturePreprocessor
import customer_churn_prediction.model as mlflow_model

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
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    help="The proportion of the dataset to include in the test split.",
)
@click.option(
    "--scale/",
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
    "--model",
    default="logreg",
    show_default=True,
    type=click.Choice(["logreg", "rf", "knn", "catboost", "lgbm", "tabnet"]),
)
def train(
    dataset_path: Path,
    random_state: int,
    test_split_ratio: float,
    scale: bool,
    ohe: bool,
    model: str,
) -> None:
    dataset = pd.read_csv(dataset_path)
    click.echo(f"Dataset shape: {dataset.shape}.")
    features = dataset.drop(TARGET_COL, axis=1)
    target = dataset[TARGET_COL]
    features_train, features_val, target_train, target_val = train_test_split(
        features, target, test_size=test_split_ratio, random_state=random_state
    )

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
        pipeline.steps.append(("num_cat_preprocessor", ct))

    classifier: mlflow_model.MLflowModel
    if model == "logreg":
        click.echo(f"Training LogisticRegression")
        classifier = mlflow_model.LogisticRegressionMLflow(
            pipeline=pipeline, random_state=random_state
        )
    elif model == "rf":
        click.echo(f"Training RandomForest")
        classifier = mlflow_model.RandomForestMLflow(
            pipeline=pipeline, random_state=random_state
        )
    elif model == "knn":
        click.echo(f"Training KNearestNeighbour")
        classifier = mlflow_model.KnnMLflow(pipeline=pipeline)
    elif model == "catboost":
        click.echo(f"Training CatBoost")
        classifier = mlflow_model.CatBoostMLflow(
            pipeline=pipeline, random_state=random_state, cat_features=CAT_COLS
        )
    elif model == "lgbm":
        click.echo(f"Training Light Gradient Boosted Machine")
        classifier = mlflow_model.LgbmMLflow(
            pipeline=pipeline, random_state=random_state
        )
    elif model == "tabnet":
        click.echo(f"Training TabNet")
        classifier = mlflow_model.TabNetMLflow(
            pipeline=pipeline, random_state=random_state
        )

    classifier.train_with_logging(features_train, target_train)
    roc_auc = classifier.evaluate(features_val, target_val)
    click.echo(f"ROC AUC score: {roc_auc}.")
