"""Endowment liquidity stress test.  Run after fetch_data.py:  python run_all.py"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src import data, model, plots  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
N, H = 10000, 120
BASE = model.Params()
ADVERSE = replace(BASE, call_rate=0.40, dist_rate=0.10, unfunded_ratio=0.60, dist_cyc=3.0)
WEIGHTS = (0.2, 0.4, 0.6)


def mc_se(p, n):
    return 1.96 * np.sqrt(max(p * (1 - p), 1e-12) / n)


def main():
    d = data.load_monthly()
    X = d[data.SLEEVES].values
    print(f"Sample {d.index[0].date()} -> {d.index[-1].date()}, {len(d)} months")
    R = model.block_bootstrap(X, N, H, block=6, seed=1)

    # 1. baseline by private weight ---------------------------------------------------------
    rows, fans = [], {}
    for w in WEIGHTS:
        p = replace(BASE, w_priv=w)
        o = model.run(R, p)
        s = model.summarise(o, p); s["w_priv"] = w
        s["ci_forced"] = mc_se(s["p_forced_cut"], N)
        rows.append(s); fans[w] = o["liq_share"]
    base = pd.DataFrame(rows).set_index("w_priv")
    base.to_csv(RES / "baseline_bootstrap.csv")
    print("\nBASELINE (bootstrap, 10y)\n", base.round(3).T.to_string())

    # 2. historical replays ------------------------------------------------------------------
    rep_rows, rep_series = [], {}
    for name, start, h in (("2008-09 crisis", "2007-11-30", 72), ("2022 rate shock", "2021-12-31", 57)):
        i = d.index.get_loc(pd.Timestamp(start)) + 1
        Rh = X[i:i + h][None]
        for label, par in (("baseline", BASE), ("adverse", ADVERSE)):
            for w in WEIGHTS:
                o = model.run(Rh, replace(par, w_priv=w))
                rep_rows.append(dict(episode=name, params=label, w_priv=w,
                                     min_liq_share=float(o["liq_share"].min()),
                                     min_cover=float(o["cover"].min()),
                                     forced=float(o["forced"].sum()), unmet=float(o["unmet"].sum()),
                                     min_real_spend=float(o["spend_real"].min()),
                                     end_real_value=float(o["V"][0, -1] / o["price_end"][0])))
                rep_series[(name, label, w)] = o["liq_share"][0]
    rep = pd.DataFrame(rep_rows)
    rep.to_csv(RES / "replays.csv", index=False)
    print("\nREPLAYS\n", rep.round(3).to_string(index=False))

    # 3. which cash-flow assumption drives it? (decomposition at 40% and 60%) ----------------
    dec = []
    for w in (0.4, 0.6):
        for label, par in (("baseline", BASE),
                           ("calls faster (40%/yr)", replace(BASE, call_rate=0.40)),
                           ("distributions slower (10%/yr)", replace(BASE, dist_rate=0.10)),
                           ("more unfunded (60%)", replace(BASE, unfunded_ratio=0.60)),
                           ("all three", replace(BASE, call_rate=0.40, dist_rate=0.10, unfunded_ratio=0.60))):
            pp = replace(par, w_priv=w); s = model.summarise(model.run(R, pp), pp)
            dec.append(dict(w_priv=w, scenario=label, p_forced=s["p_forced_cut"], p_floor=s["p_floor_breach"],
                            p_cover_lt2=s["p_cover_below_2"], median_min_cover=s["median_min_cover"]))
    dec = pd.DataFrame(dec); dec.to_csv(RES / "driver_decomposition.csv", index=False)
    print("\nDRIVERS\n", dec.round(3).to_string(index=False))

    # 4. sensitivity grid: private weight x distribution rate ----------------------------------
    ws = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    drs = [0.10, 0.15, 0.20, 0.25, 0.30]
    grid = np.zeros((len(drs), len(ws)))
    for i, dr in enumerate(drs):
        for j, w in enumerate(ws):
            pp = replace(BASE, w_priv=w, dist_rate=dr)
            grid[i, j] = model.summarise(model.run(R[:4000], pp), pp)["p_forced_cut"]
    pd.DataFrame(grid, index=[f"dist {x:.0%}" for x in drs], columns=[f"priv {x:.0%}" for x in ws]).to_csv(
        RES / "grid_forced_weight_x_dist.csv")
    print("\nGRID forced % (rows dist rate, cols private weight)\n", (pd.DataFrame(grid, index=drs, columns=ws) * 100).round(1))

    # 5. robustness ------------------------------------------------------------------------------
    rob = []
    for w in (0.4, 0.6):
        for label, par, blk in (("baseline", BASE, 6), ("block 3 months", BASE, 3), ("block 12 months", BASE, 12),
                                ("smoother private marks (0.7x vol)", replace(BASE, priv_scale=0.7), 6),
                                ("spend rate 4%", replace(BASE, spend_rate=0.04), 6),
                                ("spend rate 6%", replace(BASE, spend_rate=0.06), 6),
                                ("liquid mix 40% equity", replace(BASE, liquid_mix=(0.4, 0.4, 0.1, 0.1)), 6)):
            Rb = R if blk == 6 else model.block_bootstrap(X, N, H, block=blk, seed=1)
            pp = replace(par, w_priv=w); s = model.summarise(model.run(Rb, pp), pp)
            rob.append(dict(w_priv=w, variant=label, p_forced=s["p_forced_cut"], p_floor=s["p_floor_breach"],
                            p_cover_lt2=s["p_cover_below_2"], median_min_cover=s["median_min_cover"]))
    # the PSP proxy compounded at only ~1.9%/yr here; lift its monthly mean to the equity sleeve's mean
    lift = X[:, data.SLEEVES.index("equity")].mean() - X[:, data.SLEEVES.index("priv")].mean()
    X2 = X.copy(); X2[:, data.SLEEVES.index("priv")] += lift
    R2 = model.block_bootstrap(X2, N, H, block=6, seed=1)
    for w in (0.4, 0.6):
        pp = replace(BASE, w_priv=w); s = model.summarise(model.run(R2, pp), pp)
        rob.append(dict(w_priv=w, variant="private mean return lifted to equity mean", p_forced=s["p_forced_cut"],
                        p_floor=s["p_floor_breach"], p_cover_lt2=s["p_cover_below_2"], median_min_cover=s["median_min_cover"]))
    rob = pd.DataFrame(rob); rob.to_csv(RES / "robustness.csv", index=False)
    print("\nROBUSTNESS\n", rob.round(3).to_string(index=False))

    # 6. figures -------------------------------------------------------------------------------------
    sub_b = (f"Block bootstrap of {len(d)} monthly returns (Nov 2006-Sep 2026), {N:,} paths. "
             "Private = listed private equity proxy (PSP). Illustrative cash-flow parameters.")
    plots.fig_fan(fans, FIG / "fig1_liquid_share_fan.png", sub_b)
    plots.fig_replay({w: rep_series[("2008-09 crisis", "adverse", w)] for w in WEIGHTS}, FIG / "fig2_replay_2008_adverse.png",
                     "2008 replay: adverse cash flows (calls 40%, distributions 10%, unfunded 60%)",
                     "Actual monthly returns Dec 2007 onward. Liquid assets as % of total value.")
    plots.fig_replay({w: rep_series[("2008-09 crisis", "baseline", w)] for w in WEIGHTS}, FIG / "fig3_replay_2008_baseline.png",
                     "2008 replay: baseline cash flows (calls 25%, distributions 20%, unfunded 40%)",
                     "Actual monthly returns Dec 2007 onward. Liquid assets as % of total value.")
    plots.fig_heat(grid, ws, drs, FIG / "fig4_heatmap_weight_x_dist.png",
                   "Calls 25%/yr, unfunded 40% of private NAV; 4,000 bootstrap paths per cell.")
    line = {dr: (ws, [grid[i, j] for j in range(len(ws))]) for i, dr in enumerate(drs) if dr in (0.10, 0.20, 0.30)}
    plots.fig_prob_vs_weight(line, FIG / "fig5_cliff_by_distribution_rate.png",
                              "Calls 25%/yr, unfunded 40% of private NAV; 4,000 bootstrap paths per point. Illustrative parameters.")

    summary = dict(months=len(d), start=str(d.index[0].date()), end=str(d.index[-1].date()), paths=N, horizon_months=H,
                   baseline=base.reset_index().to_dict(orient="records"))
    (RES / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\nfigures + results written")


if __name__ == "__main__":
    main()
