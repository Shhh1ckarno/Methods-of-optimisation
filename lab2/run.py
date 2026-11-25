import json
import platform
from pathlib import Path

import highspy
import networkx as nx
import numpy as np
import pandas as pd

from src import analysis as an, plots, verify
from src.data_io import load
from src.generate_data import SEED, VARIANT, generate

ROOT = Path(__file__).parent
DATA, RES = ROOT / "data", ROOT / "results"
FIG, LOGS, SC = RES / "figures", RES / "logs", RES / "scenarios"


def main() -> None:
    for d in (FIG, LOGS, SC):
        d.mkdir(parents=True, exist_ok=True)
    generate(DATA, SEED)
    data = load(DATA)

    base = an.solve_data(data, LOGS / "solver_base.log")
    assert base.status == "optimal", base.status
    Z = base.objective
    flows = an.flows_frame(data, base.x, base.col_dual)
    flows.to_csv(RES / "flows.csv", index=False)
    pots = an.potentials_frame(data, base)
    pots.to_csv(RES / "potentials.csv", index=False)
    table, z_np = verify.check(DATA, RES / "flows.csv", RES / "potentials.csv", reported=Z)
    table.to_csv(RES / "verification.csv", index=False)

    shadow = an.capacity_shadow_table(data, base)
    shadow.to_csv(RES / "bottlenecks.csv", index=False)

    busiest = int(np.argmax(base.x))
    arc = data.arcs[busiest]
    cap_half = data.capacity.copy()
    cap_half[busiest] *= 0.5
    cap_zero = data.capacity.copy()
    cap_zero[busiest] = 0.0
    defs = [("Базовый", "base", data, {}, 1.0),
            ("Перегрузка дуги", "overload", data.with_changes(capacity=cap_half), {arc: cap_half[busiest]}, 1.0),
            ("Отказ дуги", "failure", data.with_changes(capacity=cap_zero), {arc: 0.0}, 1.0),
            ("Рост спроса", "demand", data.with_changes(demand=data.demand * 1.2), {}, 1.2)]
    rows, extra = [], {}
    for name, slug, d, caps, dem in defs:
        res = an.solve_data(d, LOGS / f"solver_{slug}.log")
        rec = {"scenario": name, "change": {"base": "нет", "overload": f"емкость {arc} -50 %",
                                             "failure": f"дуга {arc} закрыта", "demand": "нехватка в портах +20 %"}[slug],
               "status": res.status, "status_meaning": res.meaning, "objective_usd": res.objective}
        if res.status == "optimal":
            fr = an.flows_frame(d, res.x, res.col_dual)
            fr.to_csv(SC / f"{slug}_flows.csv", index=False)
            chk, _ = verify.check(DATA, SC / f"{slug}_flows.csv", reported=res.objective, demand_factor=dem, capacity=caps)
            gflow_s, unmet_s, _ = an.greedy(d)
            rec.update({"greedy_usd": float(d.cost @ gflow_s) if unmet_s <= 1e-9 else None,
                        "greedy_unmet_teu": unmet_s,
                        "greedy_gap_pct": (float(d.cost @ gflow_s) / res.objective - 1) * 100 if unmet_s <= 1e-9 else None})
            rec.update({"delta_usd": res.objective - Z, "delta_pct": (res.objective / Z - 1) * 100,
                        "verified": bool(chk.passed.all()), "saturated_arcs": " ".join(fr.arc[fr.saturated])})
            if slug == "overload":
                rec["predicted_delta_usd"] = float(base.col_dual[busiest] * (cap_half[busiest] - data.capacity[busiest]))
            if slug == "failure":
                rec["predicted_delta_usd"] = float(base.col_dual[busiest] * (0 - data.capacity[busiest]))
                plots.network(d, fr, -res.row_dual, f"Отказ дуги {arc}: новый план", FIG / "network_failure.png", [arc])
        else:
            conflict = an.conflict_set(d)
            lres, lflow, lease = an.with_leasing(d, LOGS / f"solver_{slug}_leasing.log")
            lf = an.flows_frame(d, lflow)
            lf.to_csv(SC / f"{slug}_leasing_flows.csv", index=False)
            pd.DataFrame({"port": d.ports, "lease_teu": lease}).to_csv(SC / f"{slug}_leasing.csv", index=False)
            extra[slug] = {"conflict_ports": conflict, "total_supply_teu": float(d.supply.sum()),
                           "total_demand_teu": float(d.demand.sum()), "lease_teu": float(lease.sum()),
                           "lease_by_port": {p: float(v) for p, v in zip(d.ports, lease) if v > 1e-7},
                           "cost_with_leasing_usd": lres.objective,
                           "repositioning_cost_usd": float(d.cost @ lflow),
                           "potentials_with_leasing": {p: float(v) for p, v in zip(d.ports, -lres.row_dual)}}
            rec.update({"objective_usd": None, "verified": None})
        rows.append(rec)
    sc = pd.DataFrame(rows)
    sc.to_csv(RES / "scenarios.csv", index=False)
    (RES / "infeasibility.json").write_text(json.dumps(extra, ensure_ascii=False, indent=2))

    gflow, unmet, steps = an.greedy(data)
    steps.to_csv(RES / "greedy_steps.csv", index=False)
    gf = an.flows_frame(data, gflow)
    gf.to_csv(RES / "greedy_flows.csv", index=False)
    gchk, gz = verify.check(DATA, RES / "greedy_flows.csv")
    greedy = {"greedy_cost_usd": gz, "lp_cost_usd": Z, "gap_usd": gz - Z, "gap_pct": (gz / Z - 1) * 100,
              "greedy_unmet_teu": unmet, "greedy_feasible": bool(gchk.passed.all())}

    bench = an.greedy_benchmark(generate, load, range(SEED + 1, SEED + 201))
    bench.to_csv(RES / "greedy_benchmark.csv", index=False)
    gaps = bench.gap_pct.dropna()
    greedy["benchmark"] = {"networks": int(len(bench)), "lp_feasible": int((bench.lp_status == "optimal").sum()),
                           "greedy_stuck": int((bench.greedy_unmet_teu > 1e-9).sum()),
                           "optimal_share": float((gaps < 1e-6).mean()), "mean_gap_pct": float(gaps.mean()),
                           "median_gap_pct": float(gaps.median()), "p90_gap_pct": float(gaps.quantile(0.9)),
                           "max_gap_pct": float(gaps.max())}
    plots.greedy_gaps(gaps, greedy["gap_pct"], FIG / "greedy_gaps.png")

    lp_relax, mip = an.milp_sailings(data, LOGS / "solver_milp.log")
    m = len(data.arcs)
    lp_relax.x[np.abs(lp_relax.x) < 1e-9] = 0.0
    mip.x[np.abs(mip.x) < 1e-9] = 0.0
    milp = pd.DataFrame({"arc": data.arcs, "lp_flow_teu": lp_relax.x[:m], "lp_sailings": lp_relax.x[m:],
                         "milp_flow_teu": mip.x[:m], "milp_sailings": np.round(mip.x[m:]),
                         "slots_per_sailing": data.slots, "sailing_cost_usd": data.sailing_cost})
    milp.to_csv(RES / "milp_sailings.csv", index=False)
    milp_summary = {"lp_relaxation_usd": lp_relax.objective, "milp_usd": mip.objective,
                    "gap_usd": mip.objective - lp_relax.objective, "gap_pct": (mip.objective / lp_relax.objective - 1) * 100,
                    "lp_total_sailings": float(lp_relax.x[m:].sum()), "milp_total_sailings": float(np.round(mip.x[m:]).sum()),
                    "milp_status": mip.status}

    plots.network(data, flows, -base.row_dual, "Базовый план перераспределения пустых контейнеров", FIG / "network_base.png")
    plots.scenario_costs(sc, FIG / "scenarios.png")

    summary = {"variant": VARIANT, "seed": SEED, "status": base.status, "objective_usd": Z, "objective_numpy_usd": z_np,
               "verification_passed": bool(table.passed.all()), "simplex_iterations": base.iterations,
               "busiest_arc": arc, "greedy": greedy, "milp": milp_summary,
               "python": platform.python_version(), "highs": highspy.Highs().version(), "networkx": nx.__version__}
    (RES / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
