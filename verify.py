"""Independent check: a plain-Python, one-path-at-a-time re-implementation of the engine, compared
with the vectorised model on historical replays and on 200 bootstrap paths.   python verify.py"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import data, model  # noqa: E402


def one_path(rows, p):
    """rows: list of (equity, bonds, real, priv, cash, infl) monthly returns."""
    P, L = p.w_priv * p.v0, (1 - p.w_priv) * p.v0
    U = p.unfunded_ratio * P
    S = p.spend_rate * p.v0
    S0 = S
    price = 1.0
    prices, privs = [], []
    min_share, min_real, forced, unmet = 1.0, 9e9, 0.0, 0.0
    for t, (eq, bd, re_, pv, ca, inf) in enumerate(rows):
        price *= 1 + inf
        if t >= 12 and t % 12 == 0:
            S = p.smooth * S * (1 + (price / prices[t - 12] - 1)) + (1 - p.smooth) * p.spend_rate * (P + L)
        pv *= p.priv_scale
        liq_r = p.liquid_mix[0] * eq + p.liquid_mix[1] * bd + p.liquid_mix[2] * re_ + p.liquid_mix[3] * ca
        Pg, Lg = P * (1 + pv), L * (1 + liq_r)
        tr = 0.0
        if t > 0:
            g = 1.0
            for x in privs[max(0, t - 12):t]:
                g *= 1 + x
            tr = g - 1
        da = p.dist_rate * min(max(1 + p.dist_cyc * tr, 0.3), 1.5)
        dm = 1 - (1 - min(da, 0.95)) ** (1 / 12)
        cm = 1 - (1 - p.call_rate) ** (1 / 12)
        call, dist, spend = cm * U, dm * Pg, S / 12
        Ln = Lg - call + dist - spend
        short = max(-Ln, 0.0)
        cut = min(short, spend)
        forced += short
        unmet += short - cut
        spend_act = spend - cut
        L = max(Ln, 0.0)
        P = Pg + call - dist
        U -= call
        V = P + L
        U += max(0.0, p.w_priv * V * (1 + p.unfunded_ratio) - (P + U))
        prices.append(price); privs.append(pv)
        min_share = min(min_share, L / V)
        min_real = min(min_real, spend_act * 12 / price / S0)
    return dict(min_share=min_share, min_real=min_real, forced=forced, unmet=unmet, V=V / price)


def main():
    d = data.load_monthly(); X = d[data.SLEEVES].values
    worst = 0.0
    cases = []
    for start, h in (("2007-11-30", 72), ("2021-12-31", 57)):
        i = d.index.get_loc(pd.Timestamp(start)) + 1
        cases.append(X[i:i + h])
    boot = model.block_bootstrap(X, 200, 120, seed=7)
    cases += [boot[k] for k in range(200)]
    params = [model.Params(), model.Params(call_rate=0.40, dist_rate=0.10, unfunded_ratio=0.60, dist_cyc=3.0),
              model.Params(w_priv=0.6, spend_rate=0.06)]
    n = 0
    for par in params:
        for w in (0.2, 0.4, 0.6):
            p = replace(par, w_priv=w)
            for path in cases:
                o = model.run(path[None], p)
                r = one_path([tuple(x) for x in path], p)
                diffs = [abs(o["liq_share"].min() - r["min_share"]),
                         abs(o["spend_real"].min() - r["min_real"]),
                         abs(o["forced"].sum() - r["forced"]) / 1000,
                         abs(o["V"][0, -1] / o["price_end"][0] - r["V"]) / 1000]
                worst = max(worst, max(diffs)); n += 1
    print(f"{n} path/param combinations compared; largest difference (relative units) = {worst:.2e}")
    assert worst < 1e-9, "engine and independent check disagree"
    print("PASS")


if __name__ == "__main__":
    main()
