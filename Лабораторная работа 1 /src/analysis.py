from pathlib import Path

import numpy as np
import pandas as pd

from .data_io import Data
from .model import LP, build
from .solver import Result, cost_ranges, solve

EPS = 1e-9
SIGN = {"<=": 1.0, ">=": -1.0}


def natural_rhs(lp: LP) -> np.ndarray:
    return np.array([SIGN[r.sense] * b for r, b in zip(lp.rows, lp.b)])


def natural_dual(lp: LP, res: Result) -> np.ndarray:
    return np.array([SIGN[r.sense] * d for r, d in zip(lp.rows, res.row_dual)])


def solution_frame(data: Data, lp: LP, res: Result) -> pd.DataFrame:
    rows = []
    for v, (i, j, k) in enumerate(lp.idx):
        rows.append({"supplier": data.suppliers[i], "supplier_name": data.supplier_names[i],
                     "packaging": data.packaging[j], "packaging_name": data.packaging_names[j], "tier": k + 1,
                     "qty_k": res.x[v], "tier_width_k": lp.ub[v], "unit_cost_rub_per_k": lp.c[v],
                     "cost_rub": res.x[v] * lp.c[v], "reduced_cost": res.col_dual[v]})
    return pd.DataFrame(rows)


def duals_frame(lp: LP, res: Result) -> pd.DataFrame:
    rhs, nat = natural_rhs(lp), natural_dual(lp, res)
    activity = np.array([SIGN[r.sense] * a for r, a in zip(lp.rows, lp.A @ res.x)])
    return pd.DataFrame({
        "constraint": [r.name for r in lp.rows], "group": [r.group for r in lp.rows],
        "sense": [r.sense for r in lp.rows], "unit": [r.unit for r in lp.rows], "rhs": rhs, "activity": activity,
        "slack": lp.b - lp.A @ res.x, "active": np.abs(lp.b - lp.A @ res.x) <= 1e-7, "dual_natural": nat})


def purchase_matrix(data: Data, lp: LP, y: np.ndarray) -> pd.DataFrame:
    x = np.zeros(data.offered.shape)
    for v, (i, j, _) in enumerate(lp.idx):
        x[i, j] += y[v]
    return pd.DataFrame(x, index=data.suppliers, columns=data.packaging)


def resolve_rhs(base: LP, row: int, delta_nat: float):
    b = base.b.copy()
    b[row] += SIGN[base.rows[row].sense] * delta_nat
    return solve(base.c, base.A, b, base.ub)


def shadow_table(lp: LP, res: Result) -> pd.DataFrame:
    rhs, dual = natural_rhs(lp), natural_dual(lp, res)
    Z = res.objective
    out = []
    for r, row in enumerate(lp.rows):
        if row.group == "link" or abs(dual[r]) <= EPS:
            continue
        base = abs(rhs[r])
        step = 0.01 * base if base > 0 else 1.0
        rec = {"constraint": row.name, "group": row.group, "unit": row.unit, "rhs": rhs[r],
               "shadow_rub_per_unit": dual[r], "gain_of_plus_1pct_rub": dual[r] * 0.01 * base}
        for tag, d in (("minus1pct", -step), ("plus1pct", step), ("minus20pct", -0.2 * base if base > 0 else None)):
            if d is None:
                continue
            fact = resolve_rhs(lp, r, d)
            rec[f"pred_{tag}"] = dual[r] * d
            rec[f"fact_{tag}"] = fact.objective - Z if fact.status == "optimal" else np.nan
            rec[f"status_{tag}"] = fact.status
        lo, hi = np.nan, np.nan
        if base > 0:
            tol = max(1e-6, 1e-6 * abs(dual[r]))
            for sgn in (-1, 1):
                edge, prev_val, prev = 0.0, Z, 0.0
                for t in range(1, 51):
                    d = sgn * t * step
                    f = resolve_rhs(lp, r, d)
                    if f.status != "optimal" or abs((f.objective - prev_val) / (d - prev) - dual[r]) > max(tol, 1e-4 * abs(dual[r])):
                        break
                    edge, prev_val, prev = d, f.objective, d
                if sgn < 0:
                    lo = rhs[r] + edge
                else:
                    hi = rhs[r] + edge
        rec["valid_from"], rec["valid_to"] = lo, hi
        out.append(rec)
    return pd.DataFrame(out)


def zero_explanation(data: Data, lp: LP, res: Result) -> pd.DataFrame:
    x = purchase_matrix(data, lp, res.x).to_numpy()
    rc = lp.c - lp.A.T @ res.row_dual
    out = []
    for i, s in enumerate(data.suppliers):
        for j, p in enumerate(data.packaging):
            if not data.offered[i, j]:
                out.append({"supplier": s, "packaging": p, "status": "нет предложения", "reduced_cost_rub_per_k": np.nan})
            elif x[i, j] <= 1e-7:
                vs = [v for v, t in enumerate(lp.idx) if t[:2] == (i, j)]
                w = lp.ub[vs] / lp.ub[vs].sum()
                out.append({"supplier": s, "packaging": p, "status": "не закупается",
                            "reduced_cost_rub_per_k": float(w @ rc[vs])})
    return pd.DataFrame(out)


def cost_range_table(data: Data, lp: LP, res: Result) -> pd.DataFrame:
    dn, up = cost_ranges(res)
    rows = []
    for v, (i, j, k) in enumerate(lp.idx):
        if res.x[v] > 1e-7:
            rows.append({"supplier": data.suppliers[i], "packaging": data.packaging[j], "tier": k + 1, "qty_k": res.x[v],
                         "cost_rub_per_k": lp.c[v], "cost_lower": dn[v], "cost_upper": up[v]})
    return pd.DataFrame(rows).replace({1e30: np.inf, -1e30: -np.inf})


def conflict_set(lp: LP) -> list:
    order = ["link", "quality", "share", "truck", "warehouse", "capacity", "demand"]
    keep = np.ones(len(lp.rows), bool)
    feasible = lambda k: solve(np.zeros_like(lp.c), lp.A[k], lp.b[k], lp.ub).status == "optimal"
    if feasible(keep):
        return []
    for g in order:
        for r, row in enumerate(lp.rows):
            if row.group == g:
                keep[r] = False
                if feasible(keep):
                    keep[r] = True
    return [lp.rows[r].name for r in np.nonzero(keep)[0]]


def shortfall(data: Data, lp: LP) -> pd.DataFrame:
    dem = [r for r, row in enumerate(lp.rows) if row.group == "demand"]
    S = np.zeros((len(lp.rows), len(dem)))
    for t, r in enumerate(dem):
        S[r, t] = -1.0
    A = np.hstack([lp.A, S])
    c = np.r_[np.zeros(len(lp.c)), np.ones(len(dem))]
    res = solve(c, A, lp.b, np.r_[lp.ub, np.full(len(dem), 1e7)])
    return pd.DataFrame({"packaging": data.packaging, "packaging_name": data.packaging_names,
                         "demand_k": data.demand, "shortfall_k": res.x[-len(dem):]})


def milp_exact(data: Data, lp: LP, log: Path) -> Result:
    nv = len(lp.c)
    keep = np.array([r.group != "link" for r in lp.rows])
    base = np.hstack([lp.A[keep], np.zeros((keep.sum(), nv))])
    extra = []
    for v, (i, j, k) in enumerate(lp.idx):
        a = np.zeros(2 * nv)
        a[v], a[nv + v] = 1.0, -lp.ub[v]
        extra.append(a)
        if k + 1 < data.width.shape[2]:
            a = np.zeros(2 * nv)
            a[v], a[nv + v + 1] = -1.0, lp.ub[v]
            extra.append(a)
    A = np.vstack([base, np.array(extra)])
    b = np.r_[lp.b[keep], np.zeros(len(extra))]
    return solve(np.r_[lp.c, np.zeros(nv)], A, b, np.r_[lp.ub, np.ones(nv)],
                 integrality=[False] * nv + [True] * nv, log=log)


def scan_capacity(data: Data, supplier: int, grid: np.ndarray) -> pd.DataFrame:
    rows = []
    for g in grid:
        cap = data.capacity.copy()
        cap[supplier] *= g
        lp = build(data.with_changes(capacity=cap))
        res = solve(lp.c, lp.A, lp.b, lp.ub)
        rows.append({"capacity_factor": g, "capacity_h": cap[supplier], "status": res.status, "objective": res.objective})
    return pd.DataFrame(rows)
