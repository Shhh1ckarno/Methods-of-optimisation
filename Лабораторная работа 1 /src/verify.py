import json
from pathlib import Path

import numpy as np
import pandas as pd

TOL = 1e-7


def _tensors(data_dir: Path):
    sup = pd.read_csv(data_dir / "suppliers.csv")
    pack = pd.read_csv(data_dir / "packaging.csv")
    offers = pd.read_csv(data_dir / "offers.csv")
    tiers = pd.read_csv(data_dir / "price_tiers.csv")
    S = {s: i for i, s in enumerate(sup.supplier)}
    P = {p: j for j, p in enumerate(pack.packaging)}
    m, n, K = len(S), len(P), int(tiers.tier.max())
    hours, defect = np.zeros((m, n)), np.full((m, n), np.nan)
    for r in offers.itertuples():
        hours[S[r.supplier], P[r.packaging]] = r.hours_per_k
        defect[S[r.supplier], P[r.packaging]] = r.defect_pct
    width, price = np.zeros((m, n, K)), np.zeros((m, n, K))
    for r in tiers.itertuples():
        width[S[r.supplier], P[r.packaging], r.tier - 1] = r.width_k
        price[S[r.supplier], P[r.packaging], r.tier - 1] = r.price_rub_per_k
    return sup, pack, hours, defect, width, price, S, P


def invoice_cost(x, width, price, freight, weight):
    total = float(freight @ (x @ weight))
    for i, j in zip(*np.nonzero(x > 0)):
        left = x[i, j]
        for k in range(width.shape[2]):
            take = min(left, width[i, j, k])
            total += take * price[i, j, k]
            left -= take
    return total


def check(data_dir: Path, solution_csv: Path, duals_csv: Path = None, reported=None, scale=None):
    sup, pack, hours, defect, width, price, S, P = _tensors(data_dir)
    params = json.loads((data_dir / "params.json").read_text())
    sol = pd.read_csv(solution_csv)
    y = np.zeros_like(width)
    for r in sol.itertuples():
        y[S[r.supplier], P[r.packaging], r.tier - 1] = r.qty_k
    d = pack.demand_k.to_numpy(float)
    H = sup.capacity_h.to_numpy(float)
    if scale:
        d = d * np.asarray(scale.get("demand", 1.0))
        H = H * np.asarray(scale.get("capacity", 1.0))
    w, v, qmax = pack.weight_t_per_k.to_numpy(), pack.volume_m3_per_k.to_numpy(), pack.max_defect_pct.to_numpy()
    T, f = sup.truck_t.to_numpy(float), sup.freight_rub_per_t.to_numpy(float)
    x = y.sum(axis=2)
    d_slack = np.nan_to_num(defect) - qmax
    link = []
    for k in range(y.shape[2] - 1):
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(width[:, :, k] > 0, width[:, :, k + 1] / width[:, :, k], 0.0)
        link.append(y[:, :, k] * ratio - y[:, :, k + 1])
    checks = [
        ("y >= 0", y),
        ("ширина ступени y <= u", width - y),
        ("спрос", x.sum(axis=0) - d),
        ("мощность", H - (hours * x).sum(axis=1)),
        ("логистика", T - x @ w),
        ("склад", np.array([params["warehouse_m3"] - (x @ v).sum()])),
        ("качество (ср. брак)", -(d_slack * x).sum(axis=0)),
        ("доля поставщика", params["share_max"] * d - x),
        ("порядок ступеней", np.concatenate([a.ravel() for a in link])),
    ]
    rows = [{"check": name, "min_slack": float(a.min()), "violations": int((a < -TOL).sum()),
             "passed": bool((a >= -TOL).all())} for name, a in checks]
    freight = float(f @ (x @ w))
    obj = float((price * y).sum()) + freight
    if reported is not None:
        diff = abs(obj - reported)
        rows.append({"check": "пересчёт цели, |Z_numpy - Z_solver|", "min_slack": diff, "violations": int(diff > 1e-4),
                     "passed": bool(diff <= 1e-4)})
    if duals_csv is not None:
        du = pd.read_csv(duals_csv)
        dual_obj = float((du.dual_natural * du.rhs).sum()) + float(np.minimum(sol.reduced_cost.to_numpy(), 0) @ width.ravel()[
            [(S[a] * y.shape[1] + P[b]) * y.shape[2] + c - 1 for a, b, c in zip(sol.supplier, sol.packaging, sol.tier)]])
        diff = abs(dual_obj - obj)
        rows.append({"check": "сильная двойственность, |b'lambda + u'mu - Z|", "min_slack": diff,
                     "violations": int(diff > 1e-3), "passed": bool(diff <= 1e-3)})
    return pd.DataFrame(rows), obj, invoice_cost(x, width, price, f, w), x
