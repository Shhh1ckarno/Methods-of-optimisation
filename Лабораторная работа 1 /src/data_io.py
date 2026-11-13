import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Data:
    suppliers: list
    packaging: list
    supplier_names: list
    packaging_names: list
    capacity: np.ndarray
    truck: np.ndarray
    freight: np.ndarray
    demand: np.ndarray
    weight: np.ndarray
    volume: np.ndarray
    max_defect: np.ndarray
    offered: np.ndarray
    hours: np.ndarray
    defect: np.ndarray
    width: np.ndarray
    price: np.ndarray
    warehouse: float
    share_max: float

    def with_changes(self, **kw) -> "Data":
        return replace(self, **kw)


def load(data_dir: Path) -> Data:
    sup = pd.read_csv(data_dir / "suppliers.csv")
    pack = pd.read_csv(data_dir / "packaging.csv")
    offers = pd.read_csv(data_dir / "offers.csv")
    tiers = pd.read_csv(data_dir / "price_tiers.csv")
    params = json.loads((data_dir / "params.json").read_text())
    si = {s: i for i, s in enumerate(sup["supplier"])}
    pi = {p: j for j, p in enumerate(pack["packaging"])}
    m, n, K = len(si), len(pi), int(tiers["tier"].max())
    offered, hours, defect = np.zeros((m, n), bool), np.zeros((m, n)), np.zeros((m, n))
    for r in offers.itertuples():
        i, j = si[r.supplier], pi[r.packaging]
        offered[i, j], hours[i, j], defect[i, j] = True, r.hours_per_k, r.defect_pct
    width, price = np.zeros((m, n, K)), np.zeros((m, n, K))
    for r in tiers.itertuples():
        i, j, k = si[r.supplier], pi[r.packaging], r.tier - 1
        width[i, j, k], price[i, j, k] = r.width_k, r.price_rub_per_k
    return Data(list(sup["supplier"]), list(pack["packaging"]), list(sup["name"]), list(pack["name"]),
                sup["capacity_h"].to_numpy(float), sup["truck_t"].to_numpy(float),
                sup["freight_rub_per_t"].to_numpy(float), pack["demand_k"].to_numpy(float),
                pack["weight_t_per_k"].to_numpy(float), pack["volume_m3_per_k"].to_numpy(float),
                pack["max_defect_pct"].to_numpy(float), offered, hours, defect, width, price,
                float(params["warehouse_m3"]), float(params["share_max"]))
