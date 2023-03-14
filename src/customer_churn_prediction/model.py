from abc import ABCMeta, abstractproperty
import numpy as np
from sklearn.linear_model import LogisticRegression as LogReg

class Model(metaclass=ABCMeta):
    def __init__(self, random_state=0):
        self.random_state = random_state

    @abstractproperty
    def estimator(self):
        pass

    @abstractproperty
    def params_grid(self):
        pass

class LogisticRegression(Model):
    @property
    def estimator(self):
        return LogReg(
            random_state=self.random_state
        )

    @property
    def params_grid(self):
        return {
            "model__C": np.arange(0.001, 100, 0.001),
            "model__penalty": ["l1"],
            "model__solver": ["saga"],
            "model__max_iter": [1000],
        }
