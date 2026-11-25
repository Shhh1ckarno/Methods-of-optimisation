from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from .data_io import Data
from .model import build, incidence
from .solver import Result, solve

EPS = 1e-9


def solve_data(data: Data, log: Path = None) -> Result:
    lp = build(data)
    return solve(lp.c, lp.A, lp.b, lp.ub, log=log)


def flows_frame(data: Data, flow: np.ndarray, col_dual: np.ndarray = None) -> pd.DataFrame:
    flow = np.where(np.abs(flow) < 1e-9, 0.0, flow)
    df = pd.DataFrame({"arc": data.arcs, "from": [data.ports[i] for i in data.tail], "to": [data.ports[i] for i in data.head],
                       "cost_usd_per_teu": data.cost, "capacity_teu": data.capacity, "flow_teu": flow,
                       "cost_usd": data.cost * flow, "utilization": np.divide(flow, data.capacity, out=np.zeros_like(flow),
                                                                              where=data.capacity > 0)})
    df["saturated"] = (data.capacity > 0) & (flow >= data.capacity - 1e-7)
    if col_dual is not None:
        df["capacity_shadow_usd_per_teu"] = np.where(df.saturated, col_dual, 0.0)
    return df


def potentials_frame(data: Data, res: Result) -> pd.DataFrame:
    net_in = incidence(data) @ res.x
    return pd.DataFrame({"port": data.ports, "supply_teu": data.supply, "demand_teu": data.demand,
                         "requirement_teu": data.requirement, "net_inflow_teu": net_in,
                         "slack_teu": net_in - data.requirement, "potential_usd": -res.row_dual})


def capacity_shadow_table(data: Data, base: Result, steps=(1.0, 10.0)) -> pd.DataFrame:
    rows = []
    for a in np.nonzero((base.x >= data.capacity - 1e-7) & (data.capacity > 0))[0]:
        mu = base.col_dual[a]
        rec = {"arc": data.arcs[a], "capacity_teu": data.capacity[a], "shadow_usd_per_teu": mu}
        for step in steps:
            for tag, d in (("plus", step), ("minus", -step)):
                cap = data.capacity.copy()
                cap[a] += d
                r = solve_data(data.with_changes(capacity=cap))
                rec[f"pred_{tag}{int(step)}"] = mu * d
                rec[f"fact_{tag}{int(step)}"] = r.objective - base.objective if r.status == "optimal" else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def conflict_set(data: Data) -> list:
    lp = build(data)
    keep = np.ones(len(data.ports), bool)
    feasible = lambda k: solve(np.zeros_like(lp.c), lp.A[k], lp.b[k], lp.ub).status == "optimal"
    if feasible(keep):
        return []
    for i in range(len(data.ports)):
        keep[i] = False
        if feasible(keep):
            keep[i] = True
    return [data.ports[i] for i in np.nonzero(keep)[0]]


def with_leasing(data: Data, log: Path = None):
    lp = build(data)
    n = len(data.ports)
    A = np.hstack([lp.A, -np.eye(n)])
    res = solve(np.r_[lp.c, np.full(n, data.lease)], A, lp.b, np.r_[lp.ub, np.full(n, 1e7)], log=log)
    return res, res.x[: len(data.arcs)], res.x[len(data.arcs):]


def _graph(data: Data, residual: np.ndarray) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(range(len(data.ports)))
    for a in np.nonzero(residual > EPS)[0]:
        G.add_edge(data.tail[a], data.head[a], weight=data.cost[a], arc=a)
    return G


def _push(data, G, path, supply, need, residual, flow, steps, s, d, phase):
    used = [G[u][v]["arc"] for u, v in zip(path, path[1:])]
    amount = min(supply[s], need[d], residual[used].min())
    if amount <= EPS:
        return
    flow[used] += amount
    residual[used] -= amount
    supply[s] -= amount
    need[d] -= amount
    steps.append({"step": len(steps) + 1, "phase": phase, "from": data.ports[s], "to": data.ports[d],
                  "route": "-".join(data.ports[i] for i in path), "unit_cost_usd": float(data.cost[used].sum()),
                  "amount_teu": amount})


def greedy(data: Data):
    residual = data.capacity.astype(float).copy()
    supply = np.maximum(0.0, -data.requirement)
    need = np.maximum(0.0, data.requirement)
    flow = np.zeros(len(data.arcs))
    steps = []
    G = _graph(data, residual)
    routes = []
    for s in np.nonzero(supply > EPS)[0]:
        dist, paths = nx.single_source_dijkstra(G, s, weight="weight")
        routes += [(dist[d], s, d, paths[d]) for d in np.nonzero(need > EPS)[0] if d in dist]
    for _, s, d, path in sorted(routes, key=lambda r: r[0]):
        _push(data, G, path, supply, need, residual, flow, steps, s, d, "дешевые маршруты")
    while need.sum() > EPS:
        G = _graph(data, residual)
        best = None
        for s in np.nonzero(supply > EPS)[0]:
            dist, paths = nx.single_source_dijkstra(G, s, weight="weight")
            for d in np.nonzero(need > EPS)[0]:
                if d in dist and (best is None or dist[d] < best[0]):
                    best = (dist[d], s, d, paths[d])
        if best is None:
            break
        _push(data, G, best[3], supply, need, residual, flow, steps, best[1], best[2], "довоз по остаточной сети")
    return flow, float(need.sum()), pd.DataFrame(steps)


def milp_sailings(data: Data, log: Path):
    m, n = len(data.arcs), len(data.ports)
    lp = build(data)
    trips_max = np.ceil(data.capacity / data.slots)
    A_cap = np.hstack([np.eye(m), -np.diag(data.slots)])
    A = np.vstack([np.hstack([lp.A, np.zeros((n, m))]), A_cap])
    b = np.r_[lp.b, np.zeros(m)]
    c = np.r_[data.handling, data.sailing_cost]
    ub = np.r_[data.capacity, trips_max]
    lp_relax = solve(c, A, b, ub)
    mip = solve(c, A, b, ub, integrality=[False] * m + [True] * m, log=log)
    return lp_relax, mip


def scenario_rows(name, data, res, base_z, extra=None):
    rec = {"scenario": name, "status": res.status, "status_meaning": res.meaning,
           "objective_usd": res.objective, "delta_usd": None if res.objective is None else res.objective - base_z}
    rec.update(extra or {})
    return rec


def greedy_benchmark(generate, load, seeds) -> pd.DataFrame:
    import tempfile
    rows = []
    for seed in seeds:
        with tempfile.TemporaryDirectory() as tmp:
            generate(Path(tmp), seed)
            d = load(Path(tmp))
        res = solve_data(d)
        if res.status != "optimal":
            rows.append({"seed": seed, "lp_status": res.status})
            continue
        flow, unmet, _ = greedy(d)
        g = float(d.cost @ flow)
        rows.append({"seed": seed, "lp_status": res.status, "lp_usd": res.objective, "greedy_unmet_teu": unmet,
                     "greedy_usd": g if unmet <= 1e-9 else np.nan,
                     "gap_pct": (g / res.objective - 1) * 100 if unmet <= 1e-9 else np.nan})
    return pd.DataFrame(rows)
