"""The baselines the guard has to beat at the same review budget, on the same signals.

Each score is higher for a more suspicious word. NaN in a signal that is missing counts as "nothing to
say" for the heuristics and is imputed (with an indicator) for the logistic regression.
"""

import numpy as np
from scipy.optimize import minimize

from bench.guard import metrics
from extract.guard.signals import INDEX

L2_CHOICES = (0.1, 1.0, 10.0, 100.0)


def lowest_mean_confidence(x: np.ndarray) -> np.ndarray:
    return -np.nan_to_num(x[:, INDEX["conf_mean"]], nan=1.0)


def lowest_minimum_confidence(x: np.ndarray) -> np.ndarray:
    return -np.nan_to_num(x[:, INDEX["conf_min"]], nan=1.0)


def outside_the_dictionary_first(x: np.ndarray) -> np.ndarray:
    """Words not in the dictionary first, each group ordered by lowest mean confidence."""
    outside = np.nan_to_num(1.0 - x[:, INDEX["in_dictionary"]], nan=0.0)
    return 2.0 * outside + (1.0 - np.nan_to_num(x[:, INDEX["conf_mean"]], nan=1.0))


class LogisticBaseline:
    """L2-regularised logistic regression on the standardised signals, fitted by L-BFGS."""

    def __init__(self, l2: float = 1.0) -> None:
        self.l2 = l2

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogisticBaseline":
        self.nan_columns = np.nonzero(np.isnan(x).any(axis=0))[0]
        present = ~np.isnan(x)
        self.mean = np.where(present, x, 0.0).sum(axis=0) / np.maximum(present.sum(axis=0), 1)
        design = self._impute(x)
        self.centre = design.mean(axis=0)
        self.scale = design.std(axis=0)
        self.scale[self.scale == 0] = 1.0
        design = (design - self.centre) / self.scale
        y = y.astype(np.float64)

        def loss(w):
            z = design @ w[:-1] + w[-1]
            p = 1.0 / (1.0 + np.exp(-z))
            value = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)) + 0.5 * self.l2 * np.sum(
                w[:-1] ** 2
            ) / len(y)
            gradient = np.append(design.T @ (p - y) / len(y) + self.l2 * w[:-1] / len(y), np.mean(p - y))
            return value, gradient

        start = np.zeros(design.shape[1] + 1)
        self.weights = minimize(loss, start, jac=True, method="L-BFGS-B", options={"maxiter": 500}).x
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        design = (self._impute(x) - self.centre) / self.scale
        return design @ self.weights[:-1] + self.weights[-1]

    def _impute(self, x: np.ndarray) -> np.ndarray:
        indicators = np.isnan(x[:, self.nan_columns]).astype(np.float64)
        filled = np.where(np.isnan(x), self.mean, x)
        return np.hstack([filled, indicators])


def fit_logistic(x_train, y_train, x_validation, y_validation) -> LogisticBaseline:
    """The baseline with the regularisation that gives the best validation average precision."""
    best, best_ap = None, -1.0
    for l2 in L2_CHOICES:
        model = LogisticBaseline(l2).fit(x_train, y_train)
        ap = metrics.average_precision(y_validation, model.score(x_validation))
        if ap > best_ap:
            best, best_ap = model, ap
    return best
