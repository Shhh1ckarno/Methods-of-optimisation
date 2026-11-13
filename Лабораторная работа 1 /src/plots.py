from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "axes.spines.top": False, "axes.spines.right": False})


def plan(matrix: pd.DataFrame, names: list, path: Path) -> None:
    share = matrix.div(matrix.sum(axis=0), axis=1) * 100
    fig, ax = plt.subplots(figsize=(9, 3.8))
    y = np.arange(len(names))[::-1]
    left = np.zeros(len(names))
    for s, color in zip(matrix.index, SERIES):
        vals = share.loc[s].to_numpy()
        ax.barh(y, vals, left=left, height=0.5, color=color, edgecolor=SURFACE, linewidth=2, label=s)
        for yi, l, v, q in zip(y, left, vals, matrix.loc[s].to_numpy()):
            if v >= 7:
                ax.text(l + v / 2, yi, f"{q:.0f}", ha="center", va="center", fontsize=9,
                        color=INK if color == "#eda100" else "white")
        left += vals
    ax.set_yticks(y, names)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Доля закупки вида упаковки, % (в сегментах: тыс. шт)")
    ax.set_title("Базовый план: кто и сколько поставляет", loc="left", fontsize=12)
    ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.tick_params(left=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parametric(scan: pd.DataFrame, supplier: str, base_cap: float, base_z: float, dual: float, path: Path) -> None:
    ok = scan[scan.status == "optimal"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.plot(ok.capacity_h, ok.objective / 1e6, color=SERIES[0], linewidth=2, label="Фактическая Z")
    xs = np.array([ok.capacity_h.min(), ok.capacity_h.max()])
    ax.plot(xs, (base_z + dual * (xs - base_cap)) / 1e6, color=MUTED, linewidth=1.2, linestyle="--",
            label=f"Прогноз по теневой цене ({dual:,.0f} руб/ч)".replace(",", " "))
    ax.plot([base_cap], [base_z / 1e6], "o", color=SERIES[0], markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate("базовый план", (base_cap, base_z / 1e6), textcoords="offset points", xytext=(8, 10))
    ax.set_xlabel(f"Мощность {supplier}, маш.-ч")
    ax.set_ylabel("Затраты Z, млн руб")
    ax.set_title(f"Затраты как кусочно-линейная функция мощности {supplier}", loc="left", fontsize=12)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def scenarios(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    x = np.arange(len(df))
    vals = df.objective_rub.fillna(0).to_numpy() / 1e6
    ax.bar(x, vals, width=0.5, color=SERIES[0])
    for xi, v, st in zip(x, vals, df.status):
        ax.text(xi, v + 0.1 if st == "optimal" else 0.1, f"{v:.2f}" if st == "optimal" else "нет допустимого плана",
                ha="center", color=INK if st == "optimal" else MUTED)
    ax.set_xticks(x, df.scenario)
    ax.tick_params(bottom=False)
    ax.set_ylabel("Затраты Z, млн руб")
    ax.set_title("Затраты по сценариям", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
