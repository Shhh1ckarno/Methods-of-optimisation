import json
from pathlib import Path

import numpy as np
import pandas as pd

N, Q, Y = 1, 1, 26
SEED = 10000 + 100 * Q + N
VARIANT = 1 + ((17 * N + 7 * Q + Y) % 20)
PORTS = 6
LANES = 9
LEASE_USD = 600.0
CAP_LOW, CAP_HIGH = 100, 300


def _lanes(xy: np.ndarray) -> list:
    n = len(xy)
    dist = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    inside, lanes = {0}, []
    while len(inside) < n:
        i, j = min(((a, b) for a in inside for b in range(n) if b not in inside), key=lambda p: dist[p])
        lanes.append((min(i, j), max(i, j)))
        inside.add(j)
    rest = sorted(((dist[i, j], i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in lanes))
    lanes += [(i, j) for _, i, j in rest[: LANES - len(lanes)]]
    return sorted(lanes), dist


def generate(out_dir: Path, seed: int = SEED) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    xy = np.round(rng.uniform([0, 0], [1000, 600], size=(PORTS, 2)), 0)
    order = rng.permutation(PORTS)
    surplus_ports, deficit_ports = order[:3], order[3:]
    supply, demand = np.zeros(PORTS), np.zeros(PORTS)
    demand[deficit_ports] = np.round(rng.uniform(150, 320, 3) / 10) * 10
    share = rng.dirichlet(np.full(3, 6.0))
    supply[surplus_ports] = np.round(1.12 * demand.sum() * share / 10) * 10
    ports = pd.DataFrame({"port": [f"P{i + 1}" for i in range(PORTS)], "name": [f"Порт {i + 1}" for i in range(PORTS)],
                          "x_km": xy[:, 0], "y_km": xy[:, 1], "supply_teu": supply, "demand_teu": demand})
    lanes, dist = _lanes(xy)
    arcs = []
    for i, j in lanes:
        for a, b in ((i, j), (j, i)):
            cost = round(40 + 0.22 * dist[a, b] * rng.uniform(0.85, 1.15))
            block = int(rng.choice([40, 50, 60]))
            arcs.append({"arc": f"{ports.port[a]}-{ports.port[b]}", "from": ports.port[a], "to": ports.port[b],
                         "distance_km": round(float(dist[a, b])), "cost_usd_per_teu": cost,
                         "capacity_teu": int(round(rng.uniform(CAP_LOW, CAP_HIGH) / 10) * 10),
                         "sailing_slots_teu": block, "sailing_cost_usd": round(0.6 * cost * block),
                         "handling_usd_per_teu": round(0.4 * cost, 1)})
    ports.to_csv(out_dir / "ports.csv", index=False)
    pd.DataFrame(arcs).to_csv(out_dir / "arcs.csv", index=False)
    (out_dir / "params.json").write_text(json.dumps({"N": N, "Q": Q, "Y": Y, "variant": VARIANT, "seed": seed,
                                                      "lease_usd_per_teu": LEASE_USD}, ensure_ascii=False, indent=2))
