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
    """

    def __init__(
        self,
        mlflow_tracking_uri="http://0.0.0.0:5000/",
        mlflow_experiment_name=None,
        pipeline=None,
        random_state=0,
    ):
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

    def train_with_logging(self, X, y, run_name=None):
        """
        Trains the model on the given data with logging enabled.

        Parameters:
        - X: Input data for training
        - y: Target variable for training
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
    def param_grid(self):
        """
        A dictionary of hyperparameters and their values for hyperparameter
        tuning.

        Returns:
            dict: A dictionary of hyperparameters and their values.
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
    """
    A class for training and evaluating a logistic regression model with
    MLflow autologging.

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
            The logistic regression estimator used for training and prediction.
        param_grid:
            A dictionary specifying the hyperparameter grid for hyperparameter
            tuning.

    Methods:
        train_with_logging(X, y, run_name=None):
            Train a logistic regression model with MLflow autologging enabled.
        evaluate(X_val, y_val):
            Evaluate the trained logistic regression model on a validation set.

    Examples:
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> model = LogisticRegressionMLflow(random_state=42)
        >>> model.train_with_logging(X, y)
        >>> model.evaluate(X, y)
    """

    @property
    def estimator(self):
        """
        The logistic regression estimator used for training and prediction.

        Returns:
            An instance of sklearn.linear_model.LogisticRegression.
        """
        return LogisticRegression(random_state=self.random_state)

    @property
    def param_grid(self):
        """
        A dictionary specifying the hyperparameter grid for hyperparameter
        tuning.

        Returns:
            A dictionary with keys corresponding to the hyperparameters to
            tune, and values corresponding to the ranges of values to try for
            each hyperparameter.
        """
        params = {
            "penalty": ["l1", "l2"],
            "C": np.linspace(0.1, 2, 20),
            "fit_intercept": [True, False],
            "solver": ["saga"],
            "max_iter": [500, 1000],
        }
        return {"model__" + key: val for key, val in params.items()}


class RandomForestMLflow(MLflowModel):
    """
    A class representing a random forest model trained with MLflow autologging
    enabled.

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
        estimator: sklearn.ensemble.RandomForestClassifier
            The random forest estimator used for training and prediction.
        param_grid: dict
            The parameter grid for hyperparameters tuning.


    Methods:
        train_with_logging(X, y, run_name=None):
            Train a random forest model with MLflow autologging enabled.
        evaluate(X_val, y_val):
            Evaluate the trained logistic regression model on a validation set.

    Example:
    --------
    >>> rf_model = RandomForestMLflow(random_state=42)
    >>> rf_model.train_with_logging(X_train, y_train)
    >>> rf_model.evaluate(X_test, y_test)
    """

    @property
    def estimator(self):
        return RandomForestClassifier(random_state=self.random_state)

    @property
    def param_grid(self):
        """
        A dictionary of hyperparameters and their values for hyperparameter
        tuning.

        Returns:
            dict: A dictionary of hyperparameters and their values.
        """
        params = {
            "min_samples_split": range(2, 200),
            "min_samples_leaf": range(1, 200),
            "n_estimators": [200],
        }
        return {"model__" + key: val for key, val in params.items()}
