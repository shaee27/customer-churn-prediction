from sklearn.base import BaseEstimator, TransformerMixin


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
