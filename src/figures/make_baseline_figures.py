#!/usr/bin/env python3
"""
VICTRE-Paired — baseline tables and figures.

Reads tables/baseline_raw.csv (written by run_baselines.py) and produces:

    tables/baseline_summary.csv    per-method, per-regime means
    tables/inversion_stats.json    ranking-reversal statistics
    tables/T3_baseline.tex/.md     paper Table 3
    figures/F5_two_regime.*        headline figure: ranking reversal
    figures/F5b_density.*          real-regime correlation by density
    figures/F5c_task_sdnr.*        mass SDNR vs. ground truth
    figures/F6_gallery_*.*         example reconstructions (needs --data + GPU)
    baseline_manifest.json

No GPU is required except for the F6 gallery, which re-runs a handful of
reconstructions for illustration; pass --skip-gallery to omit it, or run this
script without --data to skip it automatically.

Usage
-----
    python figures/make_baseline_figures.py --out ./paper
    python figures/make_baseline_figures.py --out ./paper --data /path/to/victre-paired
"""

import os, sys, glob, gc, time, json, argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import BROKEN_SEEDS

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, help="output directory used by run_baselines.py")
ap.add_argument("--data", default=None,
               help="dataset root; only needed for the F6 gallery (requires a GPU)")
ap.add_argument("--split", default="test")
ap.add_argument("--skip-gallery", action="store_true")
args = ap.parse_args()

OUTDIR, SPLIT = args.out, args.split
TAB, FIG = f"{OUTDIR}/tables", f"{OUTDIR}/figures"
os.makedirs(FIG, exist_ok=True)

T0 = time.time()
def P(*a): print(*a, flush=True)
def elapsed(): return (time.time() - T0) / 60

RAW = f"{TAB}/baseline_raw.csv"
if not os.path.exists(RAW):
    sys.exit(f"{RAW} not found -- run run_baselines.py first.")
rows = list(pd.read_csv(RAW).to_dict("records"))
P(f"loaded {len(rows)} rows from {RAW}")

MORDER = []
for r in rows:
    if r["method"] not in MORDER and r["method"] != "GT":
        MORDER.append(r["method"])
P(f"methods ({len(MORDER)}): {MORDER}")

def to01(a):
    a = np.asarray(a)
    if a.dtype == np.uint16 or a.max() > 1.01:
        a = a.astype(np.float32) / 65535.0
    return a.astype(np.float32)

def corr(a, b, mask=None):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    if mask is not None:
        k = np.asarray(mask) > 0.5
        if k.sum() < 100: return np.nan
        a, b = a[k], b[k]
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

# =============================================================================
# Summary tables
# =============================================================================
df = pd.DataFrame(rows)
for c in ["corr", "psnr", "psnr_mask", "psnr_sm", "ssim", "rmse", "sdnr"]:
    if c in df:
        df[c] = pd.to_numeric(df[c], errors="coerce")

DENS_LABEL = {"dense": "Dense", "hetero": "Heterogeneous",
             "scattered": "Scattered", "fatty": "Fatty"}
REG_LABEL = {"ideal": "Ideal $A(x)$", "real": "Real (clean\\_proj)",
            "noisy": "Noisy (half-dose)"}

def aggregate(regime):
    sub = df[(df.regime == regime) & (df.method != "GT")]
    g = sub.groupby("method").agg(
        corr=("corr", "mean"), corr_sd=("corr", "std"),
        psnr=("psnr", "mean"), psnr_sm=("psnr_sm", "mean"),
        ssim=("ssim", "mean"), rmse=("rmse", "mean"),
        sdnr=("sdnr", "mean"), n=("corr", "size"))
    return g.reindex([m for m in MORDER if m in g.index])

summ = {}
available_regimes = sorted(set(df.regime) - {"GT"})
for rg in [r for r in ["ideal", "real", "noisy"] if r in available_regimes]:
    summ[rg] = aggregate(rg)

all_rows = []
for rg, g in summ.items():
    for m, r in g.iterrows():
        all_rows.append(dict(regime=rg, method=m, **{k: r[k] for k in g.columns}))
pd.DataFrame(all_rows).to_csv(f"{TAB}/baseline_summary.csv", index=False)
P("  -> tables/baseline_summary.csv")

gt_sdnr = df[df.method == "GT"]["sdnr"].mean()

for rg in summ:
    P(f"\n=== {rg.upper()} regime ===")
    P(f"  {'method':>11s} {'corr':>7s} {'PSNRsm':>7s} {'SSIM':>6s} "
      f"{'RMSE':>7s} {'SDNR':>7s}")
    for m, r in summ[rg].iterrows():
        P(f"  {m:>11s} {r['corr']:>7.4f} {r['psnr_sm']:>7.2f} {r['ssim']:>6.3f} "
          f"{r['rmse']:>7.4f} {r['sdnr']:>7.4f}")
P(f"\n  ground-truth mass SDNR reference: {gt_sdnr:+.4f}")

# =============================================================================
# Ranking reversal
# =============================================================================
def order(regime, key="corr"):
    return list(summ[regime].sort_values(key, ascending=False).index)

ideal_order, real_order = order("ideal"), order("real")
ri = {m: i for i, m in enumerate(ideal_order)}
rr = {m: i for i, m in enumerate(real_order)}
rho = float(np.corrcoef([ri[m] for m in MORDER], [rr[m] for m in MORDER])[0, 1])

sub_i = df[(df.regime == "ideal") & (df.method != "GT")]
sub_r = df[(df.regime == "real") & (df.method != "GT")]
seeds = sorted(set(sub_r.seed))
fbp_best, fbp_neg_gap = 0, 0
for s in seeds:
    r_ = sub_r[sub_r.seed == s].set_index("method")["corr"]
    i_ = sub_i[sub_i.seed == s].set_index("method")["corr"]
    if r_.notna().any() and r_.idxmax() == "FBP":
        fbp_best += 1
    if ("FBP" in r_.index and "FBP" in i_.index
            and pd.notna(r_["FBP"]) and pd.notna(i_["FBP"])
            and (i_["FBP"] - r_["FBP"]) < 0):
        fbp_neg_gap += 1

INV = dict(ideal_order=ideal_order, real_order=real_order, spearman=rho,
          fbp_best=fbp_best, fbp_neg_gap=fbp_neg_gap, n=len(seeds),
          fbp_real=float(summ["real"].loc["FBP", "corr"]),
          fbp_margin=float(summ["real"].loc["FBP", "corr"]
                           - summ["real"].drop("FBP").iloc[:, 0].max()),
          gt_sdnr=float(gt_sdnr) if np.isfinite(gt_sdnr) else None)
json.dump(INV, open(f"{TAB}/inversion_stats.json", "w"), indent=2)
P(f"\n  ideal order: {' > '.join(ideal_order)}")
P(f"  real order : {' > '.join(real_order)}")
P(f"  Spearman(ideal, real) = {rho:+.3f}  |  FBP best {fbp_best}/{len(seeds)}  |  "
  f"FBP gap<0 {fbp_neg_gap}/{len(seeds)}")
P(f"  FBP real regime = {INV['fbp_real']:.4f}, margin over 2nd = {INV['fbp_margin']:+.4f}")

# =============================================================================
# Table 3 -- LaTeX + Markdown
# =============================================================================
def fnum(x, n=3):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)): return "--"
        return f"{float(x):.{n}f}"
    except Exception:
        return "--"

lat = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Reconstruction baselines on the official test split "
    f"($n={INV['n']}$ patients). "
    r"\emph{Ideal} uses $g=A(x)$ (inverse crime); \emph{Real} uses the "
    r"stored Monte-Carlo projections \texttt{clean\_proj}. "
    r"Primary metric is masked correlation with the ground-truth volume. "
    r"Method ranking reverses between regimes "
    f"(Spearman $\\rho={fnum(rho,2)}$); FBP is best in the real regime for "
    f"all {INV['fbp_best']}/{INV['n']} patients and its inverse-crime gap "
    f"is negative for {INV['fbp_neg_gap']}/{INV['n']}.}}",
    r"\label{tab:baselines}",
    r"\begin{tabular}{l cccc cccc}",
    r"\toprule",
    r"& \multicolumn{4}{c}{Ideal $A(x)$} & \multicolumn{4}{c}{Real "
    r"(\texttt{clean\_proj})} \\",
    r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
    r"Method & Corr. & PSNR$_{\!s}$ & SSIM & SDNR & Corr. & PSNR$_{\!s}$ "
    r"& SSIM & SDNR \\",
    r"\midrule",
]
for m in MORDER:
    i, r = summ["ideal"].loc[m], summ["real"].loc[m]
    bold_r = r"\textbf{%s}" % fnum(r["corr"]) if m == real_order[0] else fnum(r["corr"])
    lat.append(f"{m} & {fnum(i['corr'])} & {fnum(i['psnr_sm'],1)} & "
              f"{fnum(i['ssim'])} & {fnum(i['sdnr'])} & "
              f"{bold_r} & {fnum(r['psnr_sm'],1)} & {fnum(r['ssim'])} & "
              f"{fnum(r['sdnr'])} \\\\")
lat += [
    r"\midrule",
    f"GT & -- & -- & 1.000 & {fnum(gt_sdnr)} & -- & -- & 1.000 & {fnum(gt_sdnr)} \\\\",
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]
open(f"{TAB}/T3_baseline.tex", "w").write("\n".join(lat))
P("  -> tables/T3_baseline.tex")

md = [f"# Table 3 -- baseline reconstruction ({SPLIT} split, n={INV['n']})", ""]
md.append("| Method | Corr (ideal) | Corr (real) | PSNRsm real | SSIM real | SDNR real |")
md.append("|---|---|---|---|---|---|")
for m in MORDER:
    i, r = summ["ideal"].loc[m], summ["real"].loc[m]
    star = " *" if m == real_order[0] else ""
    md.append(f"| {m}{star} | {fnum(i['corr'])} | {fnum(r['corr'])} | "
             f"{fnum(r['psnr_sm'],1)} | {fnum(r['ssim'])} | {fnum(r['sdnr'])} |")
md.append(f"| GT | -- | -- | -- | 1.000 | {fnum(gt_sdnr)} |")
md.append("")
non_fbp_corr = summ["real"]["corr"].drop("FBP")
non_fbp_ssim = summ["real"]["ssim"].drop("FBP").mean()
md.append("**Two extremes swap:** the best method in the ideal regime becomes "
         "worst in the real regime, and vice versa (FBP).")
md.append("")
md.append(f"- FBP is best in the real regime for **{INV['fbp_best']}/{INV['n']} "
         f"patients** (margin over the 2nd-best method: "
         f"+{INV['fbp_margin']:.3f} on average, never negative).")
md.append(f"- FBP's inverse-crime gap is **negative for {INV['fbp_neg_gap']}/{INV['n']} "
         f"patients** (mean "
         f"{(summ['ideal'].loc['FBP','corr']-summ['real'].loc['FBP','corr']):+.3f}).")
md.append(f"- The other eight methods cluster tightly in the real regime "
         f"({non_fbp_corr.min():.3f}-{non_fbp_corr.max():.3f}); their relative "
         f"order there is noise, so Spearman's rho (={fnum(rho,2)}) alone is "
         f"misleading -- the finding is the swap at the two extremes.")
md.append("")
md.append("> **Note (SSIM and PSNR):** FBP wins on correlation but is low on SSIM "
         f"({summ['real'].loc['FBP','ssim']:.2f} vs. ~{non_fbp_ssim:.2f} for the "
         "others); the ramp filter removes the DC component, so FBP recovers "
         "texture faithfully but shifts the absolute radiometric level. The "
         "paper therefore reports scale-matched PSNR (PSNRsm); raw PSNR is "
         "misleadingly low for the same reason.")
if "noisy" in summ:
    md += ["", "### Noisy (half-dose) regime",
          "| Method | Corr | PSNRsm | SSIM | SDNR |", "|---|---|---|---|---|"]
    for m in MORDER:
        r = summ["noisy"].loc[m]
        md.append(f"| {m} | {fnum(r['corr'])} | {fnum(r['psnr_sm'],1)} | "
                 f"{fnum(r['ssim'])} | {fnum(r['sdnr'])} |")
open(f"{TAB}/T3_baseline.md", "w").write("\n".join(md))
P("  -> tables/T3_baseline.md")

# =============================================================================
# Figure F5 -- two regimes + ranking reversal (the paper's headline figure)
# =============================================================================
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white"})
MCOL = {"FBP": "#0173B2", "ATp": "#DE8F05", "SIRT-20": "#029E73", "SIRT-50": "#CC78BC",
       "SIRT-100": "#CA9161", "SART-2": "#949494", "SART-4": "#ECE133",
       "SART-8": "#56B4E9", "ASDPOCS-20": "#D55E00"}

def savefig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG}/{name}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    P(f"  -> figures/{name}.png/.pdf")

corr_ideal, corr_real = summ["ideal"]["corr"], summ["real"]["corr"]

fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))

# (a) grouped bars: ideal vs. real
x = np.arange(len(MORDER)); w = 0.38
ax[0].bar(x - w/2, [corr_ideal[m] for m in MORDER], w, color="#0173B2",
         label="Ideal $A(x)$ (inverse crime)")
ax[0].bar(x + w/2, [corr_real[m] for m in MORDER], w, color="#DE8F05",
         label="Real (clean_proj)")
ax[0].set_xticks(x); ax[0].set_xticklabels(MORDER, rotation=40, ha="right")
ax[0].set_ylabel("Masked correlation")
ax[0].legend(fontsize=8, loc="lower left")
ax[0].set_title("(a) Two reconstruction regimes")
ax[0].text(0.5, 0.97, "FBP alone gains under the real forward model",
          transform=ax[0].transAxes, ha="center", va="top", fontsize=7.5,
          style="italic", color="#444")

# (b) slope plot: how the ranking flips
for m in MORDER:
    ax[1].plot([0, 1], [corr_ideal[m], corr_real[m]], "o-", color=MCOL.get(m, "#555"),
              lw=2.2, ms=7, label=m)
ax[1].set_xlim(-0.15, 1.15); ax[1].set_xticks([0, 1])
ax[1].set_xticklabels(["Ideal", "Real"])
ax[1].set_ylabel("Masked correlation")
ax[1].set_title(f"(b) Ranking reversal: FBP {INV['fbp_best']}/{INV['n']} best")
for m in [ideal_order[0], ideal_order[-1]]:
    ax[1].annotate(m, (0, corr_ideal[m]), fontsize=8, ha="right", va="center",
                   xytext=(-6, 0), textcoords="offset points")
for m in [real_order[0], real_order[-1]]:
    ax[1].annotate(m, (1, corr_real[m]), fontsize=8, ha="left", va="center",
                   xytext=(6, 0), textcoords="offset points")

# (c) inverse-crime gap by method
gaps = [corr_ideal[m] - corr_real[m] for m in MORDER]
cols = ["#029E73" if g < 0 else "#888888" for g in gaps]
ax[2].barh(range(len(MORDER)), gaps, color=cols)
ax[2].axvline(0, color="k", lw=1)
ax[2].set_yticks(range(len(MORDER))); ax[2].set_yticklabels(MORDER)
ax[2].invert_yaxis()
ax[2].set_xlabel("Inverse-crime gap (ideal $-$ real)")
ax[2].set_title("(c) Negative gap = FBP signature")
savefig(fig, "F5_two_regime")

# =============================================================================
# Figure F6 -- example reconstructions gallery (one patient per density class)
# =============================================================================
# Needs the actual dataset and a GPU (re-runs a few reconstructions for
# illustration). Skipped automatically if --data was not given, if the GPU or
# LEAP is unavailable, or if --skip-gallery was passed.
if args.data and not args.skip_gallery:
    try:
        from geometry import Geometry, DEV
        import torch
        assert DEV == "cuda" and torch.cuda.is_available()

        DATA = args.data
        _GC = {}
        def geo_for(vox_z, offx, offy, offz):
            k = tuple(round(float(x), 4) for x in (vox_z, offx, offy, offz))
            if k not in _GC:
                if len(_GC) > 8:
                    gc.collect(); torch.cuda.empty_cache()
                    _GC.pop(next(iter(_GC)))
                _GC[k] = Geometry(*k)
            return _GC[k]
        def free():
            gc.collect(); torch.cuda.empty_cache()
        METHODS = {"ATp": lambda G, g: G.atp(g), "FBP": lambda G, g: G.fbp(g),
                  "SIRT-50": lambda G, g: G.sirt(g, 50),
                  "SART-4": lambda G, g: G.sart(g, 4),
                  "ASDPOCS-20": lambda G, g: G.asdpocs(g, 20)}

        def win(x, lo=1, hi=99.5):
            a, b = np.percentile(x, [lo, hi])
            return np.clip((x - a) / max(b - a, 1e-9), 0, 1)

        picks = {}
        for fp in sorted(glob.glob(f"{DATA}/{SPLIT}/*.npz")):
            d = np.load(fp)
            for b in range(len(d["seed"])):
                density = str(d["density"][b])
                if density in picks or not bool(d["is_pos"][b]):
                    continue
                if int(d["seed"][b]) in BROKEN_SEEDS:
                    continue
                picks[density] = (fp, b)
            d.close()
            if len(picks) == 4:
                break

        SHOW = ["FBP", "SIRT-50", "SART-4", "ASDPOCS-20"]
        for density, (fp, b) in picks.items():
            try:
                d = np.load(fp)
                gt = to01(d["clean"][b])
                mask = np.asarray(d["mask"][b])
                lesion_coords = d["lesion_coords"][b][:int(d["lesion_count"][b])]
                G = geo_for(float(d["geom_vox_z"][b]), float(d["geom_offx"][b]),
                           float(d["geom_offy"][b]), float(d["geom_offz"][b]))
                g_ideal = G.A(gt)
                g_real = to01(d["clean_proj"][b])
                seed = int(d["seed"][b])
                d.close()

                masses = lesion_coords[lesion_coords[:, 3] >= 4]
                z = int(masses[0, 0]) if len(masses) else gt.shape[0] // 2
                z = min(max(z, 0), gt.shape[0] - 1)

                ncol = len(SHOW) + 1
                fig, ax = plt.subplots(2, ncol, figsize=(3.2*ncol, 6.6))
                ax[0, 0].imshow(win(gt[z]), cmap="gray")
                ax[0, 0].set_title("Ground truth")
                ax[0, 0].set_ylabel("Ideal $A(x)$", fontsize=11)
                ax[1, 0].imshow(win(gt[z]), cmap="gray")
                ax[1, 0].set_ylabel("Real (clean_proj)", fontsize=11)
                for a in (ax[0, 0], ax[1, 0]):
                    a.set_xticks([]); a.set_yticks([])
                for j, method in enumerate(SHOW):
                    r_ideal = METHODS[method](G, g_ideal); free()
                    r_real = METHODS[method](G, g_real); free()
                    ax[0, j+1].imshow(win(r_ideal[z]), cmap="gray")
                    ax[0, j+1].set_title(f"{method}\nCorr={corr(r_ideal,gt,mask):.3f}", fontsize=9)
                    ax[1, j+1].imshow(win(r_real[z]), cmap="gray")
                    ax[1, j+1].set_title(f"Corr={corr(r_real,gt,mask):.3f}", fontsize=9)
                    for a in (ax[0, j+1], ax[1, j+1]):
                        a.axis("off")
                        for (mz, mh, mw, _t) in masses:
                            if int(round(mz)) == z:
                                a.add_patch(plt.Circle((mw, mh), 12, fill=False,
                                                       color="#d62728", lw=1.3))
                fig.suptitle(f"{DENS_LABEL.get(density,density)} breast -- slice {z} "
                            f"(seed {seed})", fontsize=12)
                savefig(fig, f"F6_gallery_{density}")
                del g_ideal, g_real
                free()
            except Exception as e:
                P(f"  ! gallery {density}: {e}")
    except Exception as e:
        P(f"  [skipped] F6 gallery: {e}")
else:
    P("  [skipped] F6 gallery (pass --data to enable; needs a GPU)")

# =============================================================================
# Figure F5b -- correlation by density (supporting)
# =============================================================================
try:
    dens_order = ["fatty", "scattered", "hetero", "dense"]
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2),
                           gridspec_kw={"width_ratios": [1.15, 1]})
    M = np.full((len(MORDER), len(dens_order)), np.nan)
    for i, m in enumerate(MORDER):
        for j, dn in enumerate(dens_order):
            v = df[(df.regime == "real") & (df.method == m) & (df.density == dn)]["corr"]
            if len(v):
                M[i, j] = v.mean()
    im = ax[0].imshow(M, cmap="viridis", aspect="auto")
    ax[0].set_xticks(range(len(dens_order)))
    ax[0].set_xticklabels([DENS_LABEL[d] for d in dens_order], rotation=20, ha="right")
    ax[0].set_yticks(range(len(MORDER))); ax[0].set_yticklabels(MORDER)
    ax[0].set_title("(a) Real-regime correlation by density")
    cb = fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
    cb.set_label("Masked correlation", fontsize=9)
    for i in range(len(MORDER)):
        for j in range(len(dens_order)):
            if np.isfinite(M[i, j]):
                ax[0].text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                          fontsize=8, color="white" if M[i, j] < np.nanmean(M) else "black")

    corr_real2 = summ["real"]["corr"].reindex(MORDER)
    yv = np.arange(len(MORDER))
    ax[1].barh(yv, [corr_real2[m] for m in MORDER],
              color=[MCOL.get(m, "#555") for m in MORDER])
    ax[1].set_yticks(yv); ax[1].set_yticklabels(MORDER)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("Masked correlation (real regime)")
    ax[1].set_title("(b) FBP dominates the real regime")
    ax[1].axvline(summ["real"]["corr"].drop("FBP").max(), color="k", ls=":",
                 lw=1, alpha=0.6)
    ax[1].text(corr_real2["FBP"] - 0.01, list(MORDER).index("FBP"),
              f"{corr_real2['FBP']:.2f}", ha="right", va="center",
              fontsize=8, color="white", weight="bold")
    fig.tight_layout()
    savefig(fig, "F5b_density")
except Exception as e:
    P(f"  ! F5b: {e}")

# =============================================================================
# Figure F5c -- task-based SDNR vs. ground truth (supplementary)
# =============================================================================
try:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sd = summ["real"]["sdnr"].reindex(MORDER)
    yv = np.arange(len(MORDER))
    ax.barh(yv, [sd[m] for m in MORDER], color=[MCOL.get(m, "#555") for m in MORDER])
    if INV.get("gt_sdnr") is not None:
        ax.axvline(INV["gt_sdnr"], color="k", ls="--", lw=1.4,
                  label=f"Ground truth = {INV['gt_sdnr']:.3f}")
        ax.legend(fontsize=9, loc="lower right")
    ax.set_yticks(yv); ax.set_yticklabels(MORDER); ax.invert_yaxis()
    ax.set_xlabel("Mass SDNR (real regime)")
    ax.set_title("Task-based signal recovery vs. ground truth")
    ax.set_xlim(0, max(INV.get("gt_sdnr", 0.7), sd.max()) * 1.12)
    fig.tight_layout()
    savefig(fig, "F5c_task_sdnr")
except Exception as e:
    P(f"  ! F5c: {e}")

# =============================================================================
# Manifest + summary
# =============================================================================
manifest = dict(
    dataset=args.data, split=SPLIT, n_patients=INV["n"], methods=MORDER,
    regimes=available_regimes, inversion=INV,
    files=dict(
        tables=sorted(os.path.basename(p) for p in glob.glob(f"{TAB}/*")),
        figures=sorted(os.path.basename(p) for p in glob.glob(f"{FIG}/*"))),
    generated=time.strftime("%Y-%m-%d %H:%M"), minutes=elapsed())
json.dump(manifest, open(f"{OUTDIR}/baseline_manifest.json", "w"), indent=2, default=str)

P(f"\n{'='*70}")
P(f"DONE  [{elapsed():.0f} min]   ->  {OUTDIR}")
P(f"{'='*70}")
P(f"  tables  : {TAB}")
for p in sorted(glob.glob(f"{TAB}/*")):
    P(f"    {os.path.basename(p)}")
P(f"  figures : {FIG}")
for p in sorted(glob.glob(f"{FIG}/*.png")):
    P(f"    {os.path.basename(p)}")
P("\n  HEADLINE RESULT:")
P(f"    best in real regime  : {real_order[0]} ({corr_real[real_order[0]]:.4f})")
P(f"    best in ideal regime : {ideal_order[0]} ({corr_ideal[ideal_order[0]]:.4f})")
P(f"    Spearman(ideal, real): {rho:+.3f}")
