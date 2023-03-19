from abc import ABC, abstractmethod
from catboost import CatBoostClassifier
import click
import git
import numpy as np
import lightgbm as lgb
import mlflow
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    cross_val_score,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from pytorch_tabnet.tab_model import TabNetClassifier


class MLflowModel(ABC):
    """
    A base class for creating machine learning models with MLflow support.

    Attributes:
        mlflow_tracking_uri: str
            The URI of the MLflow server to use for logging.
        mlflow_experiment_name: str
            The name of the MLflow experiment to log to.
        pipeline: Optional[Union[Pipeline, Callable[..., Pipeline]]]
            An optional pipeline for feature preprocessing.
            If provided, the model will be added to this pipeline during
            initialization.
        random_state: Optional[Union[int, RandomState]]
            An optional random state to use for reproducible results.

    Properties:
        estimator:
            The estimator used for training and prediction.
        param_grid: dict
            A dictionary specifying the hyperparameter grid for hyperparameter
            tuning.

    Methods:
        train_with_logging(X, y, run_name=None):
            Train a model with MLflow autologging enabled.
        evaluate(X_val, y_val):
            Evaluate the trained logistic regression model on a validation set.
    """

    def __init__(
        self,
        mlflow_tracking_uri="http://0.0.0.0:5000/",
        mlflow_experiment_name=None,
        pipeline=None,
        cat_features=None,
        random_state=0,
    ) -> None:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.cat_features = cat_features
        self.random_state = random_state
        self.mlflow_experiment_name = mlflow_experiment_name
        self.pipeline = pipeline if pipeline else Pipeline([])
        self.pipeline.steps.append(("model", self.estimator))

        self.name = None
        self.best_estimator = None
        self.last_run_id = None
        self.experiment_id = None

    @property
    @abstractmethod
    def estimator(self):
        """
        An estimator object that will be used for training and evaluation.

        Returns:
            object: An instance of an estimator class.
        """
        pass

    def train_with_logging(self, X, y, run_name=None) -> "MLflowModel":
        """
        Trains the model on the given data with logging enabled.

        Parameters:
            X: numpy.ndarray or pandas.DataFrame
               Training data set features.
            y: numpy.ndarray or pandas.Series
               Training data set target variable.
            run_name: str
               Name of MLflow run
        """
        mlflow.sklearn.autolog()

        repo = git.Repo(search_parent_directories=True)
        version = repo.head.object.hexsha

        with mlflow.start_run(
            experiment_id=self.experiment_id,
            run_id=self.last_run_id,
            run_name=run_name,
            tags={"mlflow.source.git.commit": version},
        ) as run:
            # store run_id and experiment_id for future calls
            self.last_run_id = run.info.run_id
            self.experiment_id = run.info.experiment_id

            # define the inner and outer cross-validation splits
            inner_cv = StratifiedKFold(
                n_splits=5, shuffle=True, random_state=self.random_state
            )
            outer_cv = StratifiedKFold(
                n_splits=5, shuffle=True, random_state=self.random_state
            )

            grid_search = GridSearchCV(
                estimator=self.pipeline,
                param_grid=self.param_grid,
                scoring="roc_auc",
                n_jobs=-1,
                cv=inner_cv,
            )

            nested_score = cross_val_score(grid_search, X=X, y=y, cv=outer_cv)
            click.echo(f"Nested CV score: {nested_score.mean():.5f}")

            self.best_estimator = grid_search.fit(X, y)

        return self

    @property
    @abstractmethod
    def param_grid(self) -> dict:
        """
        Defines the parameter grid for hyperparameter tuning.

        Returns:
            dict: A dictionary of hyperparameters and their values
            for hyperparameter tuning.
        """
        pass

    def evaluate(self, X_val, y_val) -> int:
        """
        Evaluate the performance of the trained model on a validation data set.

        Args:
            X_val: numpy.ndarray or pandas.DataFrame
                Validation data set features.
            y_val: numpy.ndarray or pandas.Series
                Validation data set target variable.

        Returns:
            int:
                R-squared score of the predictions.
        """
        if self.best_estimator is None:
            raise NotFittedError(
                "This model instance is not fitted yet. Call train with "
                "appropriate arguments before using this estimator."
            )
        with mlflow.start_run(
            experiment_id=self.experiment_id,
            run_id=self.last_run_id,
        ) as run:
            roc_auc = roc_auc_score(y_val, self.best_estimator.predict(X_val))
            mlflow.log_metric("roc_auc", roc_auc)
            return roc_auc


class LogisticRegressionMLflow(MLflowModel):
    @property
    def estimator(self):
        return LogisticRegression(random_state=self.random_state)

    @property
    def param_grid(self) -> dict:
        params = {
            "penalty": ["l1", "l2"],
            "C": np.linspace(0.1, 2, 20),
            "fit_intercept": [True, False],
            "solver": ["saga"],
            "max_iter": [500, 1000],
        }
        return {"model__" + key: val for key, val in params.items()}


class RandomForestMLflow(MLflowModel):
    @property
    def estimator(self):
        return RandomForestClassifier(random_state=self.random_state)

    @property
    def param_grid(self) -> dict:
        params = {
            "min_samples_split": range(2, 200, 20),
            "min_samples_leaf": range(1, 200, 20),
            "n_estimators": [200],
        }
        return {"model__" + key: val for key, val in params.items()}


class KnnMLflow(MLflowModel):
    @property
    def estimator(self):
        return KNeighborsClassifier()

    @property
    def param_grid(self) -> dict:
        params = {
            "n_neighbors": [44],  # range(1, 100),
            "metric": [
                "manhattan"
            ],  # ["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan", "nan_euclidean"],
        }
        return {"model__" + key: val for key, val in params.items()}


class CatBoostMLflow(MLflowModel):
    @property
    def estimator(self):
        return CatBoostClassifier(
            cat_features=self.cat_features,
            logging_level="Silent",
            eval_metric="AUC:hints=skip_train~false",
            grow_policy="Lossguide",
            metric_period=1000,
            random_seed=self.random_state,
        )

    @property
    def param_grid(self) -> dict:
        params = {
            "n_estimators": [
                250
            ],  # [5, 10, 20, 30, 40, 50, 70, 100, 150, 200, 250, 300, 500, 1000],
            "learning_rate": [
                0.05
            ],  # [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.04, 0.05, 0.1, 0.2, 0.3, 0.5],
            "max_depth": [4],  # np.arange(4, 20, 1),
            "l2_leaf_reg": [10],  # np.arange(0.1, 1, 0.05),
            "subsample": [0.6],  # [3, 5, 7, 10],
            "random_strength": [5],  # [1, 2, 5, 10, 20, 50, 100],
            "min_data_in_leaf": [100],  # np.arange(10, 1001, 10),
        }
        return {"model__" + key: val for key, val in params.items()}


class LgbmMLflow(MLflowModel):
    @property
    def estimator(self):
        return lgb.LGBMClassifier(
            verbose=-1,
            boosting_type="gbdt",
            objective="binary",
            learning_rate=0.01,
            metric="auc",
            random_state=self.random_state,
        )

    @property
    def param_grid(self) -> dict:
        params = {
            "num_leaves": [5, 7, 9, 10],
            "max_depth": [4],
            "min_child_samples": range(200, 215),
            "reg_lambda": [0, 0.1, 0.2, 0.5, 0.7, 1, 1.2, 1.5, 2],
        }
        return {"model__" + key: val for key, val in params.items()}


class TabNetMLflow(MLflowModel):
    @property
    def estimator(self):
        return TabNetClassifier(
            device_name="cpu",
            verbose=0,
            seed=self.random_state,
        )

    @property
    def param_grid(self) -> dict:
        params = {
            "gamma": [0.9, 0.92, 0.95, 0.97, 0.98],
            "lambda_sparse": [
                0.001,
                0.002,
                0.003,
                0.004,
                0.005,
                0.006,
                0.007,
                0.008,
                0.009,
                0.01,
            ],
            "momentum": np.arange(0.1, 1, 0.1),
            "n_independent": [0, 1],
            "n_shared": [7, 9],
            "n_steps": [4, 5],
        }
        return {"model__" + key: val for key, val in params.items()}


class StackingMLflow(MLflowModel):
    ESTIMATORS = ["LogReg", "KNN", "RandomForest", "CatBoost"]
    META_MODEL = CatBoostClassifier(
        logging_level="Silent",
        eval_metric="AUC:hints=skip_train~false",
        metric_period=1000,
        random_seed=0,
        grow_policy="Depthwise",
        l2_leaf_reg=1,
        learning_rate=0.08,
        max_depth=10,
        min_data_in_leaf=10,
        n_estimators=10,
        random_strength=11,
        subsample=0.1,
    )

    @property
    def estimator(self):
        return StackingClassifier(
            estimators=[
                (estim, mlflow.sklearn.load_model(f"models:/{estim}/Staging"))
                for estim in self.ESTIMATORS
            ],
            final_estimator=self.META_MODEL,
            n_jobs=-1,
        )

    def train_with_logging(self, X, y, run_name=None) -> "MLflowModel":
        mlflow.sklearn.autolog()

        repo = git.Repo(search_parent_directories=True)
        version = repo.head.object.hexsha

        with mlflow.start_run(
            experiment_id=self.experiment_id,
            run_id=self.last_run_id,
            run_name=run_name,
            tags={"mlflow.source.git.commit": version},
        ) as run:
            self.last_run_id = run.info.run_id
            self.experiment_id = run.info.experiment_id

            self.estimator.fit(X, y)

        return self

    @property
    def param_grid(self) -> dict:
        return {}
