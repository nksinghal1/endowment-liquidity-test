import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.25})
C = {0.2: "#2a9d8f", 0.4: "#e9a23b", 0.6: "#c8553d"}


def fig_replay(rep, path, title, sub):
    """rep: {w: liq_share series (H,)}, x = months from start."""
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for w, y in rep.items():
        ax.plot(np.arange(len(y)), y * 100, color=C[w], lw=2, label=f"{int(w*100)}% private")
    ax.axhline(10, color="grey", ls="--", lw=1)
    ax.text(0.5, 11, "10% liquidity floor", color="grey", fontsize=8)
    ax.set_xlabel("Months from start"); ax.set_ylabel("Liquid assets, % of total value")
    ax.set_title(title, loc="left", fontweight="bold"); fig.text(0.01, 0.01, sub, fontsize=7.5, color="#555")
    ax.legend(frameon=False); ax.set_ylim(0, 100)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(path); plt.close(fig)


def fig_prob_vs_weight(tab, path, sub):
    """tab: {dist_rate: (weights, probs)}"""
    fig, ax = plt.subplots(figsize=(8, 4.4))
    cols = {0.10: "#c8553d", 0.20: "#e9a23b", 0.30: "#2a9d8f"}
    for dr, (w, pr) in tab.items():
        ax.plot(np.array(w) * 100, np.array(pr) * 100, marker="o", color=cols.get(dr, "k"),
                lw=2, label=f"distributions {int(dr*100)}%/yr")
    ax.set_xlabel("Private allocation, % of endowment"); ax.set_ylabel("Paths with a forced liquidity event, %")
    ax.set_title("Where the liquidity cliff sits depends on distributions", loc="left", fontweight="bold")
    fig.text(0.01, 0.01, sub, fontsize=7.5, color="#555")
    ax.legend(frameon=False); fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(path); plt.close(fig)


def fig_heat(grid, xs, ys, path, sub):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    im = ax.imshow(grid * 100, origin="lower", aspect="auto", cmap="OrRd", vmin=0)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels([f"{int(x*100)}%" for x in xs])
    ax.set_yticks(range(len(ys))); ax.set_yticklabels([f"{int(y*100)}%" for y in ys])
    for i in range(len(ys)):
        for j in range(len(xs)):
            v = grid[i, j] * 100
            ax.text(j, i, f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center", va="center",
                    color="white" if v > 12 else "black", fontsize=9)
    ax.grid(False); ax.set_xlabel("Private allocation"); ax.set_ylabel("Distribution rate, per year")
    ax.set_title("% of paths with a forced liquidity event", loc="left", fontweight="bold")
    fig.text(0.01, 0.01, sub, fontsize=7.5, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(path); plt.close(fig)


def fig_fan(L_by_w, path, sub):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for w, L in L_by_w.items():
        lo, med, hi = np.percentile(L, [5, 50, 95], axis=0) * 100
        x = np.arange(L.shape[1])
        ax.plot(x, med, color=C[w], lw=2, label=f"{int(w*100)}% private (median, 5th-95th)")
        ax.fill_between(x, lo, hi, color=C[w], alpha=0.12)
    ax.axhline(10, color="grey", ls="--", lw=1)
    ax.set_xlabel("Months"); ax.set_ylabel("Liquid assets, % of total value"); ax.set_ylim(0, 100)
    ax.set_title("Liquid share of the endowment, 10-year bootstrap", loc="left", fontweight="bold")
    fig.text(0.01, 0.01, sub, fontsize=7.5, color="#555")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(path); plt.close(fig)
