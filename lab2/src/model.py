from dataclasses import dataclass

import numpy as np

from .data_io import Data


@dataclass
class LP:
    c: np.ndarray
    A: np.ndarray
    b: np.ndarray
    ub: np.ndarray


def incidence(data: Data) -> np.ndarray:
    N = np.zeros((len(data.ports), len(data.arcs)))
    N[data.head, np.arange(len(data.arcs))] = 1.0
    N[data.tail, np.arange(len(data.arcs))] = -1.0
    return N


def build(data: Data) -> LP:
    return LP(data.cost.copy(), -incidence(data), -data.requirement, data.capacity.copy())
