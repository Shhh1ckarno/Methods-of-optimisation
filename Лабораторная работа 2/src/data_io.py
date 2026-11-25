import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Data:
    ports: list
    names: list
    xy: np.ndarray
    supply: np.ndarray
    demand: np.ndarray
    arcs: list
    tail: np.ndarray
    head: np.ndarray
    cost: np.ndarray
    capacity: np.ndarray
    slots: np.ndarray
    sailing_cost: np.ndarray
    handling: np.ndarray
    lease: float

    @property
    def requirement(self) -> np.ndarray:
        return self.demand - self.supply

    def with_changes(self, **kw) -> "Data":
        return replace(self, **kw)


def load(data_dir: Path) -> Data:
    p = pd.read_csv(data_dir / "ports.csv")
    a = pd.read_csv(data_dir / "arcs.csv")
    params = json.loads((data_dir / "params.json").read_text())
    idx = {k: i for i, k in enumerate(p.port)}
    return Data(list(p.port), list(p.name), p[["x_km", "y_km"]].to_numpy(float), p.supply_teu.to_numpy(float),
                p.demand_teu.to_numpy(float), list(a.arc), a["from"].map(idx).to_numpy(), a["to"].map(idx).to_numpy(),
                a.cost_usd_per_teu.to_numpy(float), a.capacity_teu.to_numpy(float), a.sailing_slots_teu.to_numpy(float),
                a.sailing_cost_usd.to_numpy(float), a.handling_usd_per_teu.to_numpy(float),
                float(params["lease_usd_per_teu"]))
