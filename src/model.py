"""Endowment liquidity engine (Takahashi-Alexander style private-asset cash flows).

State per path: liquid assets L, private NAV P, unfunded commitments U, annual spending S.
Monthly order: returns -> capital calls and distributions -> spending -> top-up commitments.
"""
from dataclasses import dataclass, replace

import numpy as np

from .data import SLEEVES

IDX = {s: i for i, s in enumerate(SLEEVES)}


@dataclass(frozen=True)
class Params:
    w_priv: float = 0.40            # target private allocation (share of total value)
    liquid_mix: tuple = (0.60, 0.20, 0.10, 0.10)   # equity, bonds, real, cash within the liquid part
    unfunded_ratio: float = 0.40    # unfunded commitments as a share of private NAV
    call_rate: float = 0.25         # annual share of unfunded commitments called
    dist_rate: float = 0.20         # annual base distribution rate (share of private NAV)
    dist_cyc: float = 1.5           # sensitivity of distributions to trailing 12m private return
    spend_rate: float = 0.05        # target spend as share of value
    smooth: float = 0.80            # weight on prior spend (Yale-style)
    priv_scale: float = 1.0         # scale on private returns (<1 = smoother, appraisal-like marks)
    floor: float = 0.10             # liquidity floor: liquid assets / total value
    v0: float = 1000.0


def simulate(R, p: Params):
    """R: (N, H, 6) monthly sleeve returns in SLEEVES order. Returns dict of (N, H) arrays."""
    N, H, _ = R.shape
    m = np.asarray(p.liquid_mix)
    lr = (R[:, :, [IDX["equity"], IDX["bonds"], IDX["real"], IDX["cash"]]] * m).sum(-1)
    pr = R[:, :, IDX["priv"]] * p.priv_scale
    infl = R[:, :, IDX["infl"]]
    c_m = 1 - (1 - p.call_rate) ** (1 / 12)

    P = np.full(N, p.w_priv * p.v0)
    L = np.full(N, (1 - p.w_priv) * p.v0)
    U = p.unfunded_ratio * P
    S = np.full(N, p.spend_rate * p.v0)          # annual spend
    S0 = S.copy()
    price = np.ones(N)
    price_hist = np.ones((N, H))
    hist_pr = np.zeros((N, H))
    out = {k: np.zeros((N, H)) for k in
           ("V", "L", "P", "U", "spend", "spend_real", "liq_share", "cover", "call", "dist", "unmet", "forced")}

    for t in range(H):
        price = price * (1 + infl[:, t])
        if t >= 12 and t % 12 == 0:                      # annual spending-rule update
            pi12 = price / price_hist[:, t - 12] - 1
            S = p.smooth * S * (1 + pi12) + (1 - p.smooth) * p.spend_rate * (P + L)
        Pg = P * (1 + pr[:, t])
        Lg = L * (1 + lr[:, t])
        trailing = np.prod(1 + hist_pr[:, max(0, t - 12):t], axis=1) - 1 if t > 0 else np.zeros(N)
        d_ann = p.dist_rate * np.clip(1 + p.dist_cyc * trailing, 0.3, 1.5)
        d_m = 1 - (1 - np.minimum(d_ann, 0.95)) ** (1 / 12)
        call = c_m * U
        dist = d_m * Pg
        spend = S / 12
        Lnew = Lg - call + dist - spend
        short = np.maximum(-Lnew, 0)                      # cash the liquid pool cannot cover
        cut = np.minimum(short, spend)                    # first, cut spending
        unmet = short - cut                               # then fail to meet capital calls
        spend_act = spend - cut
        L = np.maximum(Lnew, 0)
        P = Pg + call - dist
        U = U - call
        V = P + L
        commit = np.maximum(0, p.w_priv * V * (1 + p.unfunded_ratio) - (P + U))
        U = U + commit
        hist_pr[:, t] = pr[:, t]
        price_hist[:, t] = price
        for k, v in (("V", V), ("L", L), ("P", P), ("U", U), ("spend", spend_act * 12),
                     ("spend_real", spend_act * 12 / price / S0), ("liq_share", L / V), ("cover", L / (S + p.call_rate * U)),
                     ("call", call), ("dist", dist), ("unmet", unmet), ("forced", short)):
            out[k][:, t] = v
    out["price_end"] = price
    return out


def summarise(o, p: Params):
    """Path-level stress metrics."""
    return {
        "p_floor_breach": float((o["liq_share"].min(1) < p.floor).mean()),
        "p_forced_cut": float((o["forced"].sum(1) > 1e-9).mean()),
        "p_unmet_call": float((o["unmet"].sum(1) > 1e-9).mean()),
        "p_cover_below_2": float((o["cover"].min(1) < 2).mean()),
        "median_min_cover": float(np.median(o["cover"].min(1))),
        "p5_min_cover": float(np.percentile(o["cover"].min(1), 5)),
        "median_min_liq_share": float(np.median(o["liq_share"].min(1))),
        "p5_min_liq_share": float(np.percentile(o["liq_share"].min(1), 5)),
        "median_end_real_value": float(np.median(o["V"][:, -1] / o["price_end"])),
        "p5_end_real_value": float(np.percentile(o["V"][:, -1] / o["price_end"], 5)),
        "median_min_real_spend": float(np.median(o["spend_real"].min(1))),
        "p5_min_real_spend": float(np.percentile(o["spend_real"].min(1), 5)),
    }


def block_bootstrap(X, N, H, block=6, seed=0):
    """Circular moving-block bootstrap of joint monthly vectors X (T, k) -> (N, H, k)."""
    rng = np.random.default_rng(seed)
    T = X.shape[0]
    nb = int(np.ceil(H / block))
    starts = rng.integers(0, T, size=(N, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % T
    idx = idx.reshape(N, nb * block)[:, :H]
    return X[idx]


def run(R, p: Params):
    return simulate(R, p)
