#!/usr/bin/env python3
"""
VICTRE-Paired — dataset characterization figures.

Reads the tables written by validate_dataset.py (tables/T4_fused.csv and,
if present, tables/T7_dose.csv, tables/T9_pedestal.csv, results.json) and
produces the paper's technical-validation figures:

    figures/F1_geometry.*      residual-parallax distribution + vs. thickness
    figures/F2_flatfield.*     air pedestal, angular uniformity, symmetry
    figures/F3_noise.*         dose-noise relationship, bit-exact histogram
    figures/F4_lesion.*        task-based SDNR separability + ROC
    figures/F6_dose.*          dose-response + reconstruction-noise correlation
                                (only if tables/T7_dose.csv exists)
    figures/F7_pedestal.*      controlled DC-pedestal sweep
                                (only if tables/T9_pedestal.csv exists)

Does not require a GPU: everything here is derived from validate_dataset.py's
CSV output, not from re-running reconstructions.

Usage
-----
    python figures/make_dataset_figures.py --validation ./validation_report
"""

import os, sys, glob, csv, json, math, argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import DENSITIES, DOSES

ap = argparse.ArgumentParser()
ap.add_argument("--validation", required=True,
               help="output directory used by validate_dataset.py (--out)")
ap.add_argument("--out", default=None,
               help="where to write figures/ (defaults to --validation)")
args = ap.parse_args()

VAL = args.validation
OUT = args.out or VAL
TAB, FIG = f"{VAL}/tables", f"{OUT}/figures"
os.makedirs(FIG, exist_ok=True)

def P(*a): print(*a, flush=True)

DENS = DENSITIES
C_MAIN = "#0173B2"
DOSE_ORDER = ["full", "half", "quarter"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white"})

def savefig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG}/{name}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    P(f"  -> figures/{name}.png/.pdf")

def read_csv(name):
    p = f"{TAB}/{name}.csv"
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))

def col(rows, key, flt=None):
    out = []
    for r in rows:
        if flt and not flt(r):
            continue
        v = r.get(key)
        if v in ("", None):
            continue
        try:
            x = float(v)
            if np.isfinite(x):
                out.append(x)
        except Exception:
            pass
    return np.array(out)

def mean(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")

def pct(a, p):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, p)) if a.size else float("nan")

RES = {}
if os.path.exists(f"{VAL}/results.json"):
    RES = json.load(open(f"{VAL}/results.json"))

F = read_csv("T4_fused")
if not F:
    sys.exit(f"{TAB}/T4_fused.csv not found -- run validate_dataset.py first.")
P(f"loaded {len(F)} rows from T4_fused.csv")

# =============================================================================
# F1 -- geometry: residual parallax
# =============================================================================
dr = col(F, "dr_max")
if len(dr):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(dr, bins=60, color=C_MAIN)
    ax[0].axvline(mean(dr), color="k", ls="--", lw=1.2, label=f"mean {mean(dr):.3f} px")
    ax[0].axvline(pct(dr, 99), color="r", ls=":", lw=1.2, label=f"99th pct {pct(dr,99):.3f}")
    ax[0].set_xlabel("|dr|max (pixels)"); ax[0].set_ylabel("patients")
    ax[0].set_title(f"Residual parallax (n={len(dr)})")
    ax[0].legend(fontsize=8)

    native_z = col(F, "native_z")
    if len(native_z) == len(dr):
        for dn in DENS:
            xs = col(F, "native_z", lambda r, d=dn: r["dens"] == d)
            ys = col(F, "dr_max", lambda r, d=dn: r["dens"] == d)
            if len(xs):
                ax[1].scatter(xs, ys, s=6, alpha=0.45, label=dn)
        ax[1].set_xlabel("native_z (breast thickness, slices)")
        ax[1].set_ylabel("|dr|max"); ax[1].legend(fontsize=7)
        ax[1].set_title("Residual parallax ~ thickness")

    xs_ = np.arange(len(DENS)); w = 0.38
    a_ = [mean(col(F, "dr_max", lambda r, d=d: r["dens"] == d)) for d in DENS]
    b_ = [mean(col(F, "dr_detrend", lambda r, d=d: r["dens"] == d)) for d in DENS]
    ax[2].bar(xs_ - w/2, a_, w, color=C_MAIN, label="|dr|max")
    ax[2].bar(xs_ + w/2, b_, w, color="#029E73", label="detrended")
    ax[2].set_xticks(xs_); ax[2].set_xticklabels(DENS, rotation=15)
    ax[2].set_ylabel("pixels"); ax[2].legend(fontsize=8)
    ax[2].set_title("By density class")
    savefig(fig, "F1_geometry")
else:
    P("  [skipped] F1_geometry (no dr_max column -- run validate_dataset.py with a GPU)")

# =============================================================================
# F2 -- flat-field / air attenuation
# =============================================================================
ap_ = col(F, "air_pedestal")
if len(ap_):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    xs_ = np.arange(len(DENS))
    t_ = [mean(col(F, "air_pedestal", lambda r, d=d: r["dens"] == d)) for d in DENS]
    ax[0].bar(xs_, t_, color=C_MAIN)
    ax[0].axhline(0, color="k", lw=1.2)
    ax[0].set_xticks(xs_); ax[0].set_xticklabels(DENS, rotation=15)
    ax[0].set_ylabel("air attenuation p (physical: 0)")
    ax[0].set_title("Flat-field pedestal")

    u = col(F, "uniformity")
    if len(u):
        ax[1].hist(u, bins=50, color=C_MAIN)
        ax[1].axvline(1 / math.cos(math.radians(25)), color="k", ls="--", lw=1.2,
                      label="1/cos25 = 1.10")
        ax[1].set_xlabel("uniformity (max/min tissue p90)")
        ax[1].legend(fontsize=8)
        ax[1].set_title("Angular uniformity")

    z = col(F, "air_zerofrac")
    if len(z):
        ax[2].hist(z, bins=50, color=C_MAIN)
        ax[2].axvline(0.5, color="k", ls="--", lw=1.2, label="expected 50%")
        ax[2].set_xlabel("fraction of exactly-zero air pixels")
        ax[2].legend(fontsize=8)
        ax[2].set_title("Air noise symmetry about zero")
    savefig(fig, "F2_flatfield")
else:
    P("  [skipped] F2_flatfield (no air_pedestal column)")

# =============================================================================
# F3 -- noise
# =============================================================================
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
any_noise = False
for i, dn in enumerate(DOSE_ORDER):
    v = col(F, f"sig_rep_{dn}")
    if len(v):
        any_noise = True
        ax[0].scatter([i] * len(v), v, s=4, alpha=0.25, color=C_MAIN)
ax[0].set_xticks(range(len(DOSE_ORDER))); ax[0].set_xticklabels(DOSE_ORDER)
ax[0].set_ylabel("sigma"); ax[0].set_title("Dose-noise relationship")

rd = np.concatenate([col(F, f"sig_reldiff_{dn}") for dn in DOSE_ORDER
                     if len(col(F, f"sig_reldiff_{dn}"))] or [np.array([0.0])])
ax[1].hist(100 * rd, bins=60, color=C_MAIN)
ax[1].set_xlabel("|measured - stored| / stored (%)")
ax[1].set_title("Stored-sigma consistency")

bd = np.concatenate([col(F, f"bitdiff_{dn}") for dn in DOSE_ORDER
                     if len(col(F, f"bitdiff_{dn}"))] or [np.array([0.0])])
ax[2].hist(bd, bins=max(3, int(bd.max()) + 2), color="#029E73")
ax[2].set_xlabel("max diff on reproduction from seed (LSB / 65535)")
ax[2].set_title("Bit-exact reproducibility")
if any_noise:
    savefig(fig, "F3_noise")
else:
    plt.close(fig)
    P("  [skipped] F3_noise (no sigma columns -- run with DO_NOISE_ARRAYS on)")

# =============================================================================
# F4 -- lesion / control ROI task-based separability
# =============================================================================
sm = col(F, "sdnr_mass"); sk = col(F, "sdnr_calc"); sc = col(F, "sdnr_ctrl")
if len(sm):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    bins = np.linspace(-2, 4, 70)
    for v, c, lbl in [(sc, "#2ca02c", "control"), (sk, "#ff7f0e", "calcification"),
                      (sm, "#d62728", "mass")]:
        if len(v):
            ax[0].hist(v, bins=bins, alpha=0.65, density=True, color=c,
                      label=f"{lbl} ({mean(v):+.3f})")
    ax[0].axvline(0, color="k", lw=1)
    ax[0].set_xlabel("SDNR"); ax[0].legend(fontsize=8)
    ax[0].set_title("Task-based separability")

    xs_ = np.arange(len(DENS))
    ax[1].bar(xs_, [mean(col(F, "sdnr_mass", lambda r, d=d: r["dens"] == d))
                    for d in DENS], color=C_MAIN)
    ax[1].set_xticks(xs_); ax[1].set_xticklabels(DENS, rotation=15)
    ax[1].set_ylabel("mass SDNR"); ax[1].set_title("Density gradient")

    if len(sc):
        th = np.linspace(min(sc.min(), sm.min()), max(sc.max(), sm.max()), 400)
        tpr = [(sm > t).mean() for t in th]
        fpr = [(sc > t).mean() for t in th]
        pooled = math.sqrt((sm.var() + sc.var()) / 2)
        dprime = (sm.mean() - sc.mean()) / max(pooled, 1e-9)
        trapz = getattr(np, "trapezoid", None) or np.trapz
        auc = float(trapz(sorted(tpr), sorted(fpr)))
        ax[2].plot(fpr, tpr, color=C_MAIN, lw=2)
        ax[2].plot([0, 1], [0, 1], "k--", lw=0.8)
        ax[2].set_xlabel("false positive rate"); ax[2].set_ylabel("true positive rate")
        ax[2].set_title(f"ROC -- mass vs. control\nd'={dprime:.3f}, AUC={abs(auc):.3f}")
    savefig(fig, "F4_lesion")
else:
    P("  [skipped] F4_lesion (no sdnr_mass column)")

# =============================================================================
# F6 -- dose response (needs tables/T7_dose.csv, from validate_dataset.py stage 7)
# =============================================================================
D = read_csv("T7_dose")
if D:
    order = ["clean"] + DOSE_ORDER
    methods = []
    for r in D:
        if r["method"] not in methods:
            methods.append(r["method"])
    mcol = {"FBP": "#0173B2", "SIRT-50": "#CC78BC"}

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for m in methods:
        cs = [mean(col(D, "corr", lambda r, m=m, o=o: r["method"] == m and r["dose"] == o))
              for o in order]
        sds = [mean(col(D, "mass_sdnr", lambda r, m=m, o=o: r["method"] == m and r["dose"] == o))
              for o in order]
        ax[0].plot(range(len(order)), cs, "o-", color=mcol.get(m), label=m, lw=2)
        ax[1].plot(range(len(order)), sds, "o-", color=mcol.get(m), label=m, lw=2)
    for a in ax[:2]:
        a.set_xticks(range(len(order))); a.set_xticklabels(order, rotation=15)
        a.legend(fontsize=8)
    ax[0].set_ylabel("masked correlation"); ax[0].set_title("Dose response")
    ax[1].set_ylabel("mass SDNR"); ax[1].set_title("Task performance vs. dose")

    for m in methods:
        zs = [mean(col(D, "recon_lag1_z", lambda r, m=m, o=o: r["method"] == m and r["dose"] == o))
              for o in order[1:]]
        ax[2].plot(range(len(order) - 1), zs, "o-", color=mcol.get(m), label=f"{m} (z)", lw=2)
    ax[2].axhline(0, color="k", ls="--", lw=0.8)
    ax[2].set_xticks(range(len(order) - 1)); ax[2].set_xticklabels(order[1:], rotation=15)
    ax[2].set_ylabel("reconstruction-noise lag-1 (z-axis)")
    ax[2].set_title("Reconstruction noise is CORRELATED along z\n"
                    "(adding white noise to the volume would be wrong)")
    ax[2].legend(fontsize=8)
    savefig(fig, "F6_dose")
else:
    P("  [skipped] F6_dose (tables/T7_dose.csv not found -- run validate_dataset.py "
      "with stage 7 enabled)")

# =============================================================================
# F7 -- controlled DC-pedestal sweep (needs tables/T9_pedestal.csv)
# =============================================================================
PE = read_csv("T9_pedestal")
if PE:
    methods = []
    for r in PE:
        if r["method"] not in methods:
            methods.append(r["method"])
    peds = sorted({float(r["pedestal"]) for r in PE})
    mcol = {"FBP": "#0173B2", "ATp": "#DE8F05", "SIRT-20": "#029E73", "SIRT-50": "#CC78BC"}

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for m in methods:
        ys = [mean(col(PE, "corr", lambda r, m=m, p=p: r["method"] == m
                       and abs(float(r["pedestal"]) - p) < 1e-9)) for p in peds]
        zs = [mean(col(PE, "self_corr", lambda r, m=m, p=p: r["method"] == m
                       and abs(float(r["pedestal"]) - p) < 1e-9)) for p in peds]
        ax[0].plot(peds, ys, "o-", color=mcol.get(m), label=m, lw=2)
        ax[1].plot(peds, zs, "o-", color=mcol.get(m), label=m, lw=2)
    ax[0].set_xlabel("injected DC offset delta"); ax[0].set_ylabel("masked correlation")
    ax[0].set_title("Reconstruction quality vs. DC pedestal"); ax[0].legend(fontsize=8)
    ax[1].axhline(1.0, color="k", ls="--", lw=0.8)
    ax[1].set_xlabel("injected DC offset delta")
    ax[1].set_ylabel("correlation with the delta=0 output")
    ax[1].set_title("Blindness to DC (1.0 = fully immune)"); ax[1].legend(fontsize=8)
    savefig(fig, "F7_pedestal")
else:
    P("  [skipped] F7_pedestal (tables/T9_pedestal.csv not found -- run validate_dataset.py "
      "with stage 9 enabled)")

P("\nDone.")
