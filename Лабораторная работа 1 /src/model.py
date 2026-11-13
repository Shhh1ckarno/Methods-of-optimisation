from dataclasses import dataclass

import numpy as np

from .data_io import Data


@dataclass
class Row:
    name: str
    group: str
    sense: str
    unit: str


@dataclass
class LP:
    c: np.ndarray
    A: np.ndarray
    b: np.ndarray
    ub: np.ndarray
    rows: list
    idx: list


def build(data: Data, link_tiers: bool = True) -> LP:
    m, n = len(data.suppliers), len(data.packaging)
    K = data.width.shape[2]
    idx = [(i, j, k) for i in range(m) for j in range(n) for k in range(K) if data.offered[i, j]]
    pos = {t: v for v, t in enumerate(idx)}
    nv = len(idx)
    c = np.array([data.price[t] + data.freight[t[0]] * data.weight[t[1]] for t in idx])
    ub = np.array([data.width[t] for t in idx])

    def pair(i, j):
        return [pos[(i, j, k)] for k in range(K)] if data.offered[i, j] else []

    A, b, rows = [], [], []

    def add(coefs, rhs, name, group, sense, unit):
        a = np.zeros(nv)
        for v, val in coefs:
            a[v] += val
        s = -1.0 if sense == ">=" else 1.0
        A.append(s * a)
        b.append(s * rhs)
        rows.append(Row(name, group, sense, unit))

    for j in range(n):
        add([(v, 1.0) for i in range(m) for v in pair(i, j)], data.demand[j],
            f"Спрос {data.packaging[j]}", "demand", ">=", "тыс. шт")
    for i in range(m):
        add([(v, data.hours[i, j]) for j in range(n) for v in pair(i, j)], data.capacity[i],
            f"Мощность {data.suppliers[i]}", "capacity", "<=", "маш.-ч")
    for i in range(m):
        add([(v, data.weight[j]) for j in range(n) for v in pair(i, j)], data.truck[i],
            f"Логистика {data.suppliers[i]}", "truck", "<=", "т")
    add([(v, data.volume[j]) for i in range(m) for j in range(n) for v in pair(i, j)], data.warehouse,
        "Склад", "warehouse", "<=", "м3")
    for j in range(n):
        add([(v, data.defect[i, j] - data.max_defect[j]) for i in range(m) for v in pair(i, j)], 0.0,
            f"Качество {data.packaging[j]}", "quality", "<=", "%·тыс. шт")
    for i in range(m):
        for j in range(n):
            if data.offered[i, j]:
                add([(v, 1.0) for v in pair(i, j)], data.share_max * data.demand[j],
                    f"Доля {data.suppliers[i]}-{data.packaging[j]}", "share", "<=", "тыс. шт")
    if link_tiers:
        for i in range(m):
            for j in range(n):
                if data.offered[i, j]:
                    for k in range(K - 1):
                        ratio = data.width[i, j, k + 1] / data.width[i, j, k]
                        add([(pos[(i, j, k + 1)], 1.0), (pos[(i, j, k)], -ratio)], 0.0,
                            f"Ступени {data.suppliers[i]}-{data.packaging[j]}-{k + 1}", "link", "<=", "тыс. шт")
    return LP(c, np.array(A), np.array(b), ub, rows, idx)
