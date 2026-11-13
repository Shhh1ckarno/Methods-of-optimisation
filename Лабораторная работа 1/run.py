#!/usr/bin/env python
import json
import platform
from pathlib import Path

import highspy
import numpy as np
import pandas as pd

from src import analysis as an, plots, verify
from src.data_io import load
from src.generate_data import SEED, VARIANT, generate
from src.model import build
from src.solver import solve

ROOT = Path(__file__).parent
DATA, RES = ROOT / "data", ROOT / "results"
FIG, LOGS, SC = RES / "figures", RES / "logs", RES / "scenarios"


def main() -> None:
    for d in (FIG, LOGS, SC):
        d.mkdir(parents=True, exist_ok=True)
    generate(DATA, SEED)
    data = load(DATA)

    lp = build(data)
    base = solve(lp.c, lp.A, lp.b, lp.ub, log=LOGS / "solver_base.log")
    assert base.status == "optimal", base.status
    Z = base.objective
    sol = an.solution_frame(data, lp, base)
    sol.to_csv(RES / "solution.csv", index=False)
    duals = an.duals_frame(lp, base)
    duals.to_csv(RES / "duals.csv", index=False)
    matrix = an.purchase_matrix(data, lp, base.x)
    matrix.to_csv(RES / "purchase_matrix.csv")

    table, z_np, invoice, _ = verify.check(DATA, RES / "solution.csv", RES / "duals.csv", reported=Z)
    table.to_csv(RES / "verification.csv", index=False)

    shadow = an.shadow_table(lp, base)
    shadow.to_csv(RES / "shadow_vs_actual.csv", index=False)
    an.cost_range_table(data, lp, base).to_csv(RES / "cost_ranges.csv", index=False)
    an.zero_explanation(data, lp, base).to_csv(RES / "zero_pairs.csv", index=False)

    cap = shadow[shadow.group == "capacity"].copy()
    order = cap.reindex(cap.shadow_rub_per_unit.abs().sort_values(ascending=False).index)
    top = [data.suppliers.index(c.split()[-1]) for c in order.constraint]
    top1, top2 = top[0], top[1]

    m = len(data.suppliers)
    def caps(**idx):
        c = np.ones(m)
        for i, v in idx.items():
            c[int(i[1:])] = v
        return c
    defs = [("Базовый", np.ones(m), 1.0),
            ("Дефицитный", caps(**{f"i{top1}": 0.8}), 1.0),
            ("Стрессовый", caps(**{f"i{top1}": 0.8, f"i{top2}": 0.8}), 1.05),
            ("Критический", caps(**{f"i{top1}": 0.8, f"i{top2}": 0.8}), 1.10)]
    row_index = {r.name: i for i, r in enumerate(lp.rows)}
    rhs = an.natural_rhs(lp)
    dual = an.natural_dual(lp, base)
    rows, conflicts = [], {}
    for name, cf, df in defs:
        d = data.with_changes(capacity=data.capacity * cf, demand=data.demand * df)
        slug = {"Базовый": "base", "Дефицитный": "deficit", "Стрессовый": "stress", "Критический": "critical"}[name]
        lps = build(d)
        res = solve(lps.c, lps.A, lps.b, lps.ub, log=LOGS / f"solver_{slug}.log")
        rec = {"scenario": name, "capacity_factors": " ".join(f"{v:g}" for v in cf), "demand_factor": df,
               "status": res.status, "status_meaning": res.meaning}
        pred = Z
        for j, p in enumerate(data.packaging):
            pred += dual[row_index[f"Спрос {p}"]] * data.demand[j] * (df - 1.0)
        for i, s in enumerate(data.suppliers):
            pred += dual[row_index[f"Мощность {s}"]] * data.capacity[i] * (cf[i] - 1.0)
        if res.status == "optimal":
            rec["predicted_by_shadow_rub"] = pred
        if res.status == "optimal":
            frame = an.solution_frame(d, lps, res)
            csv = SC / f"{slug}_solution.csv"
            frame.to_csv(csv, index=False)
            chk, z_s, inv, _ = verify.check(DATA, csv, reported=res.objective, scale={"capacity": cf, "demand": df})
            mat = an.purchase_matrix(d, lps, res.x)
            mat.to_csv(SC / f"{slug}_matrix.csv")
            rec.update({"objective_rub": res.objective, "delta_to_base_rub": res.objective - Z,
                        "delta_pct": (res.objective / Z - 1) * 100, "invoice_rub": inv, "verified": bool(chk.passed.all()),
                        **{f"qty_{s}": float(mat.loc[s].sum()) for s in data.suppliers}})
        else:
            conflicts[name] = {"iis": an.conflict_set(lps)}
            sf = an.shortfall(d, lps)
            sf.to_csv(SC / f"{slug}_shortfall.csv", index=False)
            conflicts[name]["total_shortfall_k"] = float(sf.shortfall_k.sum())
            rec.update({"verified": None})
        rows.append(rec)
    sc = pd.DataFrame(rows)
    sc.to_csv(RES / "scenarios.csv", index=False)
    (RES / "conflicts.json").write_text(json.dumps(conflicts, ensure_ascii=False, indent=2))

    naive = build(data, link_tiers=False)
    nv = solve(naive.c, naive.A, naive.b, naive.ub)
    pd.DataFrame(an.solution_frame(data, naive, nv)).to_csv(RES / "naive_solution.csv", index=False)
    _, _, nv_inv, _ = verify.check(DATA, RES / "naive_solution.csv")
    milp = an.milp_exact(data, lp, LOGS / "solver_milp.log")
    milp_y = milp.x[:len(lp.c)]
    milp_frame = an.solution_frame(data, lp, an.Result("optimal", "", milp.objective, milp_y, np.zeros_like(milp_y), None, 0, milp.highs))
    milp_frame.to_csv(RES / "milp_solution.csv", index=False)
    milp_chk, _, milp_inv, _ = verify.check(DATA, RES / "milp_solution.csv", reported=milp.objective)
    comparison = pd.DataFrame([
        {"model": "ЛП без связи ступеней (некорректная)", "status": nv.status, "model_cost_rub": nv.objective, "invoice_cost_rub": nv_inv},
        {"model": "ЛП с линейной связью ступеней (основная)", "status": base.status, "model_cost_rub": Z, "invoice_cost_rub": invoice},
        {"model": "MILP, точные ступени", "status": milp.status, "model_cost_rub": milp.objective, "invoice_cost_rub": milp_inv},
    ])
    comparison.to_csv(RES / "model_comparison.csv", index=False)

    scan = an.scan_capacity(data, top1, np.round(np.arange(0.4, 1.81, 0.02), 2))
    scan.to_csv(RES / "parametric_capacity.csv", index=False)
    s1 = data.suppliers[top1]
    plots.plan(matrix, data.packaging_names, FIG / "plan.png")
    plots.parametric(scan, s1, float(data.capacity[top1]), Z, float(dual[row_index[f"Мощность {s1}"]]), FIG / "parametric.png")
    plots.scenarios(sc, FIG / "scenarios.png")

    summary = {
        "variant": VARIANT, "seed": SEED, "status": base.status, "status_meaning": base.meaning, "objective_rub": Z,
        "objective_numpy_rub": z_np, "invoice_rub": invoice, "milp_rub": milp.objective, "milp_verified": bool(milp_chk.passed.all()),
        "simplex_iterations": base.iterations, "verification_passed": bool(table.passed.all()),
        "n_vars": len(lp.c), "n_rows": len(lp.rows), "active_rows": int(duals.active.sum()),
        "active_with_zero_dual": int((duals.active & (duals.dual_natural.abs() <= 1e-9)).sum()),
        "zero_vars": int((base.x <= 1e-7).sum()), "top_capacity_supplier": s1,
        "python": platform.python_version(), "highs": highspy.Highs().version(),
    }
    (RES / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
