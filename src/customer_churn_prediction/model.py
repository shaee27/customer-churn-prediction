from abc import ABC, abstractmethod
import git
import numpy as np
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


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
        random_state=0,
    ) -> None:
        self.random_state = random_state
        self.mlflow_tracking_uri = mlflow_tracking_uri
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
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
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

            grid_search = GridSearchCV(
                estimator=self.pipeline,
                param_grid=self.param_grid,
                scoring="roc_auc",
                n_jobs=-1,
                cv=10,
                refit=True,
            )
            self.best_estimator = grid_search.fit(X, y)

        return self

    @property
    @abstractmethod
    def param_grid(self) -> dict:
        """
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
