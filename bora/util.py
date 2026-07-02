"""Contains utility functions."""

import json
from typing import List

import numpy as np


class NotUniqueError(Exception):
    """A point is non-unique."""


class ConstraintNotSupportedError(Exception):
    """Raised when constrained optimization is not supported."""


class NoValidPointRegisteredError(Exception):
    """Raised when an acquisition function depends on previous points
    but none are registered."""


class TargetSpaceEmptyError(Exception):
    """Raised when the target space is empty."""


class IneqConstraint:

    def __init__(self, c, keys: List[str]):
        self.c = c
        self.keys = keys

    def to_dict(self, x):
        return {key: x[i] for i, key in enumerate(self.keys)}

    def fun_minus_lb(self, x):
        return self.c(self.to_dict(x)) - self.c["lb"]

    def ub_minus_fun(self, x):
        return self.c["ub"] - self.c(self.to_dict(x))


class ExperimentConstraint:

    def __init__(self, experiment_constraint, keys: List[str]):
        self.experiment_constraint = experiment_constraint
        self.keys = keys

    def to_dict(self, x):
        return {key: x[i] for i, key in enumerate(self.keys)}

    def fun_minus_lb(self, x):
        return (self.experiment_constraint.eval(**self.to_dict(x)) -
                self.experiment_constraint.lb)

    def ub_minus_fun(self, x):
        return self.experiment_constraint.ub - self.experiment_constraint.eval(
            **self.to_dict(x))


def load_logs(optimizer, logs):
    """Load previous ...

    Parameters
    ----------
    optimizer : BayesianOptimizer
        Optimizer the register the previous observations with.

    logs : str or bytes or os.PathLike
        File to load the logs from.

    Returns
    -------
    The optimizer with the state loaded.

    """
    if isinstance(logs, str):
        logs = [logs]

    for log in logs:
        with open(log, "r") as j:
            while True:
                try:
                    iteration = next(j)
                except StopIteration:
                    break

                iteration = json.loads(iteration)
                try:
                    optimizer.register(
                        params=iteration["params"],
                        target=iteration["target"],
                        constraint_value=(iteration["constraint"] if
                                          optimizer.is_constrained else None),
                    )
                except NotUniqueError:
                    continue

    return optimizer


def ensure_rng(random_state=None):
    """Create a random number generator based on an optional seed.

    Parameters
    ----------
    random_state : np.random.RandomState or int or None, default=None
        Random state to use. if `None`, will create an unseeded random state.
        If `int`, creates a state using the argument as seed. If a
        `np.random.RandomState` simply returns the argument.

    Returns
    -------
    np.random.RandomState

    """
    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        assert isinstance(random_state, np.random.RandomState)
    return random_state

#Converts a point representation into an immutable tuple which is hashable so it can be used as a dictionary key
#A dict key must be hashable, and numpy arrays are not
#Original force converted every element to floats, doesn't work with strings
def hashable(x):
    if isinstance(x, dict):
        return tuple(x[k] for k in sorted(x))

    if isinstance(x, np.ndarray):
        return tuple(x.tolist())

    return tuple(x)

def add_samples_to_cache(samples, cache):
    for x in samples:
        cache[hashable(x.ravel())] = None
