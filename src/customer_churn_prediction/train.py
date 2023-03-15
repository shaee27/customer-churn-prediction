from pathlib import Path

import click
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import FeaturePreprocessor
from .model import MLflowModel, LogisticRegressionMLflow, RandomForestMLflow

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
)
@click.option(
    "--model",
    default="logreg",
    show_default=True,
    type=click.Choice(["logreg", "rf"]),
)
def train(
    dataset_path: Path, random_state: int, test_split_ratio: float, model: str
) -> None:
    dataset = pd.read_csv(dataset_path)
    click.echo(f"Dataset shape: {dataset.shape}.")
    features = dataset.drop(TARGET_COL, axis=1)
    target = dataset[TARGET_COL]
    features_train, features_val, target_train, target_val = train_test_split(
        features, target, test_size=test_split_ratio, random_state=random_state
    )

    pipeline = Pipeline(
        steps=[
            ("feature_preprocessor", FeaturePreprocessor()),
            (
                "num_cat_preprocessor",
                ColumnTransformer(
                    transformers=[
                        ("num", StandardScaler(), NUM_COLS),
                        ("cat", OneHotEncoder(), CAT_COLS),
                    ]
                ),
            ),
        ]
    )
    classifier: MLflowModel
    if model == "logreg":
        click.echo(f"Training LogisticRegression")
        classifier = LogisticRegressionMLflow(
            pipeline=pipeline, random_state=random_state
        )
    elif model == "rf":
        click.echo(f"Training RandomForest")
        classifier = RandomForestMLflow(
            pipeline=pipeline, random_state=random_state
        )
    classifier.train_with_logging(features_train, target_train)
    roc_auc = classifier.evaluate(features_val, target_val)
    click.echo(f"ROC AUC score: {roc_auc}.")
