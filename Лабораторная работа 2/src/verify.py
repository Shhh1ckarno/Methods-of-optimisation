import json
from pathlib import Path

import numpy as np
import pandas as pd

TOL = 1e-7


def check(data_dir: Path, flows_csv: Path, potentials_csv: Path = None, reported=None,
          demand_factor: float = 1.0, capacity: dict = None):
    ports = pd.read_csv(data_dir / "ports.csv")
    arcs = pd.read_csv(data_dir / "arcs.csv")
    flows = pd.read_csv(flows_csv).set_index("arc").loc[arcs.arc]
    idx = {p: i for i, p in enumerate(ports.port)}
    f = flows.flow_teu.to_numpy(float)
    u = arcs.capacity_teu.to_numpy(float).copy()
    for name, value in (capacity or {}).items():
        u[arcs.arc.tolist().index(name)] = value
    c = arcs.cost_usd_per_teu.to_numpy(float)
    req = ports.demand_teu.to_numpy(float) * demand_factor - ports.supply_teu.to_numpy(float)
    net_in = np.zeros(len(ports))
    for a, (t, h) in enumerate(zip(arcs["from"], arcs["to"])):
        net_in[idx[h]] += f[a]
        net_in[idx[t]] -= f[a]
    rows = [("поток не меньше 0", f), ("поток не больше пропускной способности", u - f),
            ("баланс в каждом порту", net_in - req)]
    out = [{"check": n, "min_slack": float(v.min()), "violations": int((v < -TOL).sum()),
            "passed": bool((v >= -TOL).all())} for n, v in rows]
    z = float(c @ f)
    if reported is not None:
        out.append({"check": "пересчет цели", "min_slack": abs(z - reported), "violations": int(abs(z - reported) > 1e-6),
                    "passed": abs(z - reported) <= 1e-6})
    if potentials_csv is not None:
        pi = pd.read_csv(potentials_csv).set_index("port").loc[ports.port].potential_usd.to_numpy(float)
        t = arcs["from"].map(idx).to_numpy()
        h = arcs["to"].map(idx).to_numpy()
        rc = c + pi[t] - pi[h]
        mu = np.maximum(0.0, -rc)
        cs = np.where(f <= TOL, rc, np.where(f >= u - TOL, -rc, -np.abs(rc)))
        node_cs = -np.abs(pi * (net_in - req))
        dual = float(pi @ req - u @ mu)
        out += [{"check": "потенциалы не меньше 0", "min_slack": float(pi.min()), "violations": int((pi < -TOL).sum()),
                 "passed": bool((pi >= -TOL).all())},
                {"check": "дополняющая нежесткость по дугам", "min_slack": float(cs.min()),
                 "violations": int((cs < -1e-6).sum()), "passed": bool((cs >= -1e-6).all())},
                {"check": "дополняющая нежесткость по портам", "min_slack": float(node_cs.min()),
                 "violations": int((node_cs < -1e-6).sum()), "passed": bool((node_cs >= -1e-6).all())},
                {"check": "двойственная цель равна прямой", "min_slack": abs(dual - z),
                 "violations": int(abs(dual - z) > 1e-6), "passed": abs(dual - z) <= 1e-6}]
    return pd.DataFrame(out), z
