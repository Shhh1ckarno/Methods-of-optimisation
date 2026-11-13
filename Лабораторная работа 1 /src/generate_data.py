import json
from pathlib import Path

import numpy as np
import pandas as pd

N, Q, Y = 1, 1, 26
SEED = 10000 + 100 * Q + N
VARIANT = 1 + ((17 * N + 7 * Q + Y) % 20)
SHARE_MAX = 0.5
TIERS = 3

PACKAGING = [
    ("P1", "Гофрокороб", 120, 32000, 0.25, 9.0, 2.0),
    ("P2", "Картонный лоток", 200, 9000, 0.06, 2.4, 2.5),
    ("P3", "Термоплёночный пакет", 300, 2600, 0.012, 0.25, 3.0),
    ("P4", "Бумажный пакет", 250, 5000, 0.02, 0.5, 2.5),
]
BASE_HOURS = [0.9, 0.5, 0.15, 0.25]


def generate(out_dir: Path, seed: int = SEED) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    m, n = 5, len(PACKAGING)
    pack = pd.DataFrame(PACKAGING, columns=["packaging", "name", "demand_k", "base_price",
                                            "weight_t_per_k", "volume_m3_per_k", "max_defect_pct"])
    d = pack["demand_k"].to_numpy(float)
    hours_total = float(np.dot(BASE_HOURS, d))
    weight_total = float(np.dot(pack["weight_t_per_k"], d))
    volume_total = float(np.dot(pack["volume_m3_per_k"], d))

    cap_shares = rng.dirichlet(np.full(m, 12.0))
    truck_shares = rng.dirichlet(np.full(m, 12.0))
    sup = pd.DataFrame({
        "supplier": [f"S{i + 1}" for i in range(m)],
        "name": [f"Поставщик {c}" for c in "АБВГД"],
        "capacity_h": np.round(1.2 * hours_total * cap_shares, 1),
        "truck_t": np.round(1.4 * weight_total * truck_shares, 2),
        "freight_rub_per_t": np.round(rng.uniform(1500, 4500, m), 0),
    })
    price_factor = rng.uniform(0.90, 1.12, m)
    missing = {(int(rng.integers(m)), j) for j in range(n)}

    offers, tiers = [], []
    for i in range(m):
        for j in range(n):
            if (i, j) in missing:
                continue
            q_max = pack.loc[j, "max_defect_pct"]
            f = price_factor[i] * rng.uniform(0.97, 1.03)
            defect = q_max * max(0.4, 0.8 + 3.0 * (1.0 - f) + rng.normal(0, 0.12))
            offers.append({"supplier": f"S{i + 1}", "packaging": f"P{j + 1}",
                           "hours_per_k": round(BASE_HOURS[j] * rng.uniform(0.8, 1.25), 4),
                           "defect_pct": round(defect, 3)})
            total = d[j] * rng.uniform(0.55, 0.80)
            widths = np.round(total * np.array([0.30, 0.40, 0.30]), 1)
            price = pack.loc[j, "base_price"] * f
            for k in range(TIERS):
                tiers.append({"supplier": f"S{i + 1}", "packaging": f"P{j + 1}", "tier": k + 1,
                              "width_k": float(widths[k]), "price_rub_per_k": round(float(price))})
                price *= 1.0 - rng.uniform(0.04, 0.08)

    sup.to_csv(out_dir / "suppliers.csv", index=False)
    pack.drop(columns="base_price").to_csv(out_dir / "packaging.csv", index=False)
    pd.DataFrame(offers).to_csv(out_dir / "offers.csv", index=False)
    pd.DataFrame(tiers).to_csv(out_dir / "price_tiers.csv", index=False)
    (out_dir / "params.json").write_text(json.dumps({
        "N": N, "Q": Q, "Y": Y, "variant": VARIANT, "seed": seed,
        "warehouse_m3": round(1.12 * volume_total, 1), "share_max": SHARE_MAX,
    }, ensure_ascii=False, indent=2))
