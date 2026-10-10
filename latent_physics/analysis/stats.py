"""Small statistics shared by the analyses."""
from __future__ import annotations

import numpy as np


def r2(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    return 1 - ((y - yhat) ** 2).sum(0) / (((y - y.mean(0)) ** 2).sum(0) + 1e-12)



def corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
