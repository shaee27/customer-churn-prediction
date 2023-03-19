import click
import pandas as pd
from pathlib import Path
from typing import Tuple

from sklearn.base import BaseEstimator, TransformerMixin

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

TARGET_COL = "Churn"


class FeaturePreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, copy=None):
        X = X.copy()
        X = X.replace(
            {
                " ": 0,
                "No": 0,
                "No internet service": 0,
                "No phone service": 0,
                "Yes": 1,
                "Male": 0,
                "Female": 1,
                "DSL": 1,
                "Fiber optic": 2,
                "Month-to-month": 0,
                "One year": 1,
                "Two year": 2,
                "Credit card (automatic)": 0,
                "Bank transfer (automatic)": 1,
                "Mailed check": 2,
                "Electronic check": 3,
            }
        )
        X["TotalSpent"] = X.TotalSpent.astype(float)
        return X


def get_dataset(
    csv_path: Path, random_state: int
) -> Tuple[pd.DataFrame, pd.Series]:
    dataset = pd.read_csv(csv_path)
    click.echo(f"Dataset shape: {dataset.shape}.")
    features = dataset.drop(TARGET_COL, axis=1)
    target = dataset[TARGET_COL]

    return features, target
