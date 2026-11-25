from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#c3c2b7", "#ffffff"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": SURFACE})


def layout(data) -> np.ndarray:
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(len(data.ports)))
    G.add_edges_from(zip(data.tail, data.head))
    geo = (data.xy - data.xy.mean(axis=0)) / data.xy.std()
    pos = nx.spring_layout(G, pos={i: geo[i] for i in G}, k=1.2, iterations=200, seed=10101)
    xy = np.array([pos[i] for i in range(len(data.ports))])
    xy -= xy.min(axis=0)
    return xy / xy.max(axis=0) * np.array([1000.0, 620.0])


def network(data, flows: pd.DataFrame, potentials, title: str, path: Path, closed: list = ()) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    xy = layout(data)
    center = xy.mean(axis=0)
    fmax = max(flows.flow_teu.max(), 1.0)
    for a, row in flows.reset_index(drop=True).iterrows():
        p0, p1 = xy[data.tail[a]], xy[data.head[a]]
        rad = 0.16
        f = row.flow_teu
        if row.arc in closed:
            color, width, style = RED, 1.2, (0, (2, 2))
        elif f > 1e-7:
            color, width, style = (RED if row.saturated else BLUE), 1.2 + 5.0 * f / fmax, "solid"
        else:
            color, width, style = GRID, 1.0, (0, (4, 3))
        ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=14,
                                     color=color, linewidth=width, linestyle=style, shrinkA=22, shrinkB=22, zorder=1))
        mid, d = (p0 + p1) / 2, p1 - p0
        ctrl = mid + rad * np.array([d[1], -d[0]])
        lab = 0.25 * p0 + 0.5 * ctrl + 0.25 * p1
        text = "закрыта" if row.arc in closed else f"{f:.0f}/{row.capacity_teu:.0f}"
        ax.text(*lab, text, fontsize=8.5, ha="center", va="center", color=INK if f > 1e-7 else MUTED,
                bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.9), zorder=3)
    for i, port in enumerate(data.ports):
        r = data.requirement[i]
        color = AQUA if r < 0 else ORANGE
        ax.scatter(*xy[i], s=900, color=color, edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.text(*xy[i], port, ha="center", va="center", color="white", fontsize=10, fontweight="bold", zorder=5)
        info = f"{'избыток' if r < 0 else 'нужно'} {abs(r):.0f}"
        if potentials is not None:
            info += f"\nπ = {potentials[i]:.0f}"
        away = xy[i] - center
        away = away / (np.linalg.norm(away) + 1e-9)
        ax.text(*(xy[i] + away * 85), info, ha="center", va="center", fontsize=9, color=MUTED, zorder=5)
    ax.set_xlim(-140, 1140)
    ax.set_ylim(-130, 750)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=12)
    handles = [Line2D([], [], marker="o", ls="", color=AQUA, markersize=11, label="порт с избытком"),
               Line2D([], [], marker="o", ls="", color=ORANGE, markersize=11, label="порт с нехваткой"),
               Line2D([], [], color=BLUE, lw=3, label="поток, TEU (поток/емкость)"),
               Line2D([], [], color=RED, lw=3, label="насыщенная дуга"),
               Line2D([], [], color=GRID, lw=1.5, ls=(0, (4, 3)), label="дуга без потока")]
    ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def scenario_costs(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    x = np.arange(len(df))
    vals = df.objective_usd.fillna(0).to_numpy() / 1000
    ax.bar(x, vals, width=0.5, color=BLUE)
    for xi, v, st, lease in zip(x, vals, df.status, df.get("lease_total_usd", pd.Series([np.nan] * len(df)))):
        label = f"{v:.1f}" if st == "optimal" else "нет плана"
        ax.text(xi, v + 1 if st == "optimal" else 1, label, ha="center", color=INK if st == "optimal" else MUTED)
    ax.set_xticks(x, df.scenario)
    ax.tick_params(bottom=False)
    ax.set_ylabel("Затраты, тыс. долл.")
    ax.set_title("Затраты на перераспределение по сценариям", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def greedy_gaps(gaps: pd.Series, own_gap: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    bins = np.arange(0, max(gaps.max(), 1) + 2.5, 2.5)
    ax.hist(gaps, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=2)
    ax.axvline(own_gap, color=INK, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.annotate("вариант 11", (own_gap, ax.get_ylim()[1] * 0.9), xytext=(8, 0), textcoords="offset points", color=INK)
    ax.set_xlabel("Зазор жадной эвристики к оптимуму ЛП, %")
    ax.set_ylabel("Число сетей")
    ax.set_title(f"Жадная эвристика на {len(gaps)} случайных сетях", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
