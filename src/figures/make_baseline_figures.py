# ═══════════════════════════════════════════════════════════════════════════
# make_baseline_figures.py — baseline tables and figures (9 methods incl. FBP)
#
#   Reads rows_test_final.npy (produced by run_baselines.py) and produces the
#   paper-ready tables and two figures. No GPU needed.
#
#   Metric hierarchy used throughout (see run_baselines.py docstring for why):
#     corr_mask (breast-masked correlation)  — PRIMARY cross-regime metric
#     psnr_scaled (scale-matched PSNR)       — corroborating metric
#     sdnr (mass SDNR)                       — task-based corroboration
#     corr (whole-volume) and bg (background energy) — diagnostics only
#
#   Output figures:
#     fig4_baseline_regimes.pdf — 4 panels: (a) SIRT corr_mask vs iterations,
#       (b) corr_mask by method ordered by gap, (c) inverse-crime gap (both
#       correlation and scale-matched PSNR) vs reliance on the forward model,
#       (d) mass SDNR by method.
#     fig6_psnr.png — raw vs scale-matched PSNR, 9 methods.
#   Saves to Drive paper_assets/baseline_v6 (+ local backup, with retry).
# ═══════════════════════════════════════════════════════════════════════════
import os, glob, time, shutil
import numpy as np, pandas as pd

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
except Exception as e:
    print("drive.mount skipped:", e)

PAPER_DIR = "/content/drive/MyDrive/New_DBT/paper_assets/baseline_v6"
LOCAL_DIR = "/content/baseline_results/paper"
for d in (LOCAL_DIR, PAPER_DIR):
    try: os.makedirs(d, exist_ok=True)
    except Exception as e: print(f"  mkdir ({d}):", e)

def _push(lp):
    dst = os.path.join(PAPER_DIR, os.path.basename(lp))
    for i in range(5):
        try: shutil.copy2(lp, dst); return dst
        except Exception as e:
            if i == 4: print(f"  ⚠️ Drive copy failed {os.path.basename(lp)} (safe locally): {e}"); return None
            time.sleep(3)

def save_fig(fig, fname, pdf=False):
    lp = os.path.join(LOCAL_DIR, fname)
    fig.savefig(lp, dpi=300, bbox_inches="tight", facecolor="white"); _push(lp)
    if pdf:
        lpp = lp.replace(".png", ".pdf"); fig.savefig(lpp, bbox_inches="tight", facecolor="white"); _push(lpp)
    return lp

def save_text(t, fname):
    lp = os.path.join(LOCAL_DIR, fname)
    with open(lp, "w") as fh: fh.write(t)
    _push(lp); return lp

def find(name):
    for p in [f"/content/baseline_results/{name}", os.path.join(PAPER_DIR, name),
              os.path.join(LOCAL_DIR, name)]:
        if os.path.exists(p): return p
    hits = sorted(glob.glob(f"/content/baseline_results/**/{name}", recursive=True))
    return hits[0] if hits else None

# ── load ───────────────────────────────────────────────────────────────────
p_rows = find("rows_test_final.npy")
assert p_rows, "rows_test_final.npy not found — run run_baselines.py first"
print(f"✅ rows: {p_rows}")
dm = pd.DataFrame(list(np.load(p_rows, allow_pickle=True)))

gt = dm[dm.method == "GT"].copy()
main = dm[dm.method != "GT"].copy()

# Order by inverse-crime gap (ascending): FBP relies least on the forward
# operator (single filtered back-projection, no data-consistency loop) and
# SIRT-100 relies most (100 enforcements); the gap grows monotonically along
# this same order (see Table 5).
ORDER = [m for m in ["FBP", "ATp", "SART-2", "SART-4", "SART-8",
                     "ASDPOCS-20", "SIRT-20", "SIRT-50", "SIRT-100"]
         if m in main.method.unique()]
N_PAT = main.seed.nunique()
DENS_LABEL = {"scattered": "Scattered", "hetero": "Heterogeneous", "dense": "Dense", "fatty": "Fatty"}
print(f"   {N_PAT} patients | methods: {ORDER}")

# ── style ────────────────────────────────────────────────────────────────
import matplotlib; matplotlib.use("Agg")
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import Patch
mpl.rcParams.update({
    "savefig.dpi": 300, "figure.dpi": 120, "font.family": "DejaVu Sans",
    "font.size": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white", "axes.axisbelow": True,
})
C_IDEAL, C_REAL, C_GAP = "#0173B2", "#DE8F05", "#029E73"

def piv(col, sub=None):
    d = sub if sub is not None else main
    m = d.pivot_table(index="method", columns="regime", values=col, aggfunc="mean").reindex(ORDER)
    s = d.pivot_table(index="method", columns="regime", values=col, aggfunc="std").reindex(ORDER)
    return m, s

cm_m, cm_s = piv("corr_mask")     # PRIMARY
c_m, c_s   = piv("corr")          # whole-volume, diagnostic
ps_m, ps_s = piv("psnr_scaled")
pr_m, pr_s = piv("psnr_mask")
bg_m, bg_s = piv("bg")
sd_m, sd_s = piv("sdnr", sub=main[main.ip])

def iters_of(m):
    if m in ("FBP", "ATp"): return 1
    try: return int(m.split("-")[1])
    except Exception: return np.nan

# ════════════════════════════════════════════════════════════════════════
# FIGURE 4 — 4-panel baseline figure (primary metric: corr_mask)
# ════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(2, 2, figsize=(12.5, 10))
x = np.arange(len(ORDER)); w = 0.38

# (a) SIRT correlation (masked) vs iterations
sirt = [m for m in ORDER if m.startswith("SIRT-")]
xi = [iters_of(m) for m in sirt]
ax[0,0].errorbar(xi, [cm_m.loc[m,"ideal"] for m in sirt], yerr=[cm_s.loc[m,"ideal"] for m in sirt],
                 marker="o", ms=6, lw=2, color=C_IDEAL, capsize=3)
ax[0,0].errorbar(xi, [cm_m.loc[m,"real"] for m in sirt], yerr=[cm_s.loc[m,"real"] for m in sirt],
                 marker="s", ms=6, lw=2, color=C_REAL, capsize=3)
ax[0,0].annotate("improves", xy=(xi[-1], cm_m.loc[sirt[-1],"ideal"]), xytext=(-4,8),
                 textcoords="offset points", color=C_IDEAL, fontsize=9, ha="right")
ax[0,0].annotate("degrades", xy=(xi[-1], cm_m.loc[sirt[-1],"real"]), xytext=(-4,-14),
                 textcoords="offset points", color=C_REAL, fontsize=9, ha="right")
ax[0,0].set_xlabel("SIRT iterations"); ax[0,0].set_ylabel("Correlation (breast mask)")
ax[0,0].set_title("(a) Iterative refinement under the two regimes")
ax[0,0].set_xticks(xi)

# (b) correlation (masked) by method, ordered by gap
ax[0,1].bar(x-w/2, [cm_m.loc[m,"ideal"] for m in ORDER], w, yerr=[cm_s.loc[m,"ideal"] for m in ORDER],
           color=C_IDEAL, capsize=2)
ax[0,1].bar(x+w/2, [cm_m.loc[m,"real"] for m in ORDER], w, yerr=[cm_s.loc[m,"real"] for m in ORDER],
           color=C_REAL, capsize=2)
ax[0,1].set_xticks(x); ax[0,1].set_xticklabels(ORDER, rotation=35, ha="right")
ax[0,1].set_ylabel("Correlation (breast mask)"); ax[0,1].set_title("(b) Correlation by method")

# (c) inverse-crime gap (correlation, left axis) + scale-matched PSNR gap (right axis)
gap_corr = (cm_m["ideal"] - cm_m["real"]).reindex(ORDER)
gap_psnr = (ps_m["ideal"] - ps_m["real"]).reindex(ORDER)
axc2 = ax[1,0].twinx()
ax[1,0].bar(x, gap_corr.values, color=C_GAP, alpha=0.85)
axc2.plot(x, gap_psnr.values, "k^--", ms=7, lw=1.6)
ax[1,0].set_xticks(x); ax[1,0].set_xticklabels(ORDER, rotation=35, ha="right")
ax[1,0].set_ylabel("Correlation gap (ideal − real)", color=C_GAP)
axc2.set_ylabel("Scale-matched PSNR gap (dB)")
ax[1,0].tick_params(axis="y", labelcolor=C_GAP)
ax[1,0].set_title("(c) Inverse-crime gap grows with reliance on the forward model")
ax[1,0].annotate("no iteration", xy=(0, gap_corr.iloc[0]), xytext=(0, 0.02),
                 fontsize=8.5, color=C_GAP)
ax[1,0].annotate(f"{iters_of(ORDER[-1])} iterations", xy=(len(ORDER)-1, gap_corr.iloc[-1]),
                 xytext=(len(ORDER)-4.3, gap_corr.iloc[-1]+0.02), fontsize=8.5, color=C_GAP)

# (d) mass SDNR by method
ax[1,1].bar(x-w/2, [sd_m.loc[m,"ideal"] for m in ORDER], w, yerr=[sd_s.loc[m,"ideal"] for m in ORDER],
           color=C_IDEAL, capsize=2)
ax[1,1].bar(x+w/2, [sd_m.loc[m,"real"] for m in ORDER], w, yerr=[sd_s.loc[m,"real"] for m in ORDER],
           color=C_REAL, capsize=2)
gt_sdnr = gt.sdnr.dropna()
ax[1,1].axhline(gt_sdnr.mean(), color="k", ls="--", lw=1.2)
ax[1,1].set_xticks(x); ax[1,1].set_xticklabels(ORDER, rotation=35, ha="right")
ax[1,1].set_ylabel("Mass SDNR"); ax[1,1].set_title("(d) Lesion conspicuity")

handles = [Patch(color=C_IDEAL, label="Ideal $A$(clean) — inverse crime"),
           Patch(color=C_REAL, label="Real Monte Carlo — inverse-crime-free"),
           Patch(color=C_GAP, label="Gap (ideal − real), correlation"),
           plt.Line2D([0],[0], color="k", ls="--", marker="^", label="Gap, scale-matched PSNR"),
           plt.Line2D([0],[0], color="k", ls="--", lw=1.2, label=f"Ground-truth SDNR ({gt_sdnr.mean():+.2f})")]
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06))
fig.tight_layout(rect=[0,0,1,0.94])
print("✅ Figure 4:", save_fig(fig, "fig4_baseline_regimes.png", pdf=True)); plt.close(fig)

# ════════════════════════════════════════════════════════════════════════
# FIGURE 6 — raw vs scale-matched PSNR
# ════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
for k, (mm, ssd, ttl) in enumerate([(pr_m, pr_s, "(a) Raw masked PSNR"),
                                    (ps_m, ps_s, "(b) Scale-matched PSNR")]):
    ax2[k].bar(x-w/2, [mm.loc[m,"ideal"] for m in ORDER], w, yerr=[ssd.loc[m,"ideal"] for m in ORDER],
              color=C_IDEAL, capsize=2)
    ax2[k].bar(x+w/2, [mm.loc[m,"real"] for m in ORDER], w, yerr=[ssd.loc[m,"real"] for m in ORDER],
              color=C_REAL, capsize=2)
    ax2[k].set_xticks(x); ax2[k].set_xticklabels(ORDER, rotation=35, ha="right"); ax2[k].set_title(ttl)
ax2[0].set_ylabel("PSNR (dB)")
fig2.legend(handles=[Patch(color=C_IDEAL, label="Ideal $A$(clean)"),
                    Patch(color=C_REAL, label="Real Monte Carlo")],
           loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.1))
fig2.tight_layout(rect=[0,0,1,0.92])
print("✅ Figure 6:", save_fig(fig2, "fig6_psnr.png", pdf=True)); plt.close(fig2)

# ════════════════════════════════════════════════════════════════════════
# TABLES
# ════════════════════════════════════════════════════════════════════════
try: import tabulate  # noqa
except Exception:
    import subprocess, sys
    subprocess.run(f"{sys.executable} -m pip install -q tabulate", shell=True)

def tbl(col, prec=3, signed=False, sub=None):
    m, s = piv(col, sub); sg = "+" if signed else ""
    out = pd.DataFrame(index=ORDER)
    for reg, lab in [("ideal", "Ideal A(clean)"), ("real", "Real MC")]:
        out[lab] = [f"{m.loc[k,reg]:{sg}.{prec}f} ± {s.loc[k,reg]:.{prec}f}" for k in ORDER]
    return out

corrMaskT = tbl("corr_mask", 3)
corrT     = tbl("corr", 3)
psnrRawT  = tbl("psnr_mask", 2)
psnrSclT  = tbl("psnr_scaled", 2)
sdnrT     = tbl("sdnr", 3, signed=True, sub=main[main.ip])
bgT       = tbl("bg", 4)

gap_c = (cm_m["ideal"] - cm_m["real"]).reindex(ORDER)
gap_p = (ps_m["ideal"] - ps_m["real"]).reindex(ORDER)
gapT = pd.DataFrame({"Correlation gap": [f"{gap_c[m]:.3f}" for m in ORDER],
                     "Scale-matched PSNR gap (dB)": [f"{gap_p[m]:.2f}" for m in ORDER]}, index=ORDER)

gt_bg  = gt.bg.dropna().mean()
gt_sdnr_txt = f"{gt_sdnr.mean():+.3f} ± {gt_sdnr.std():.3f}"

md = f"""# VICTRE-Paired — Baseline reconstruction results (test split, n={N_PAT} patients)

Nine reconstructors are evaluated under two projection regimes: **ideal** `A(clean)` (synthetic
forward projections computed by the same operator that inverts them — an inverse crime by
construction, Kaipio & Somersalo 2005) and **real** (the independent Monte-Carlo projections,
inverse-crime-free). Reconstructions are min–max normalized to [0,1]. SEED 208084664 is excluded.

FBP is implemented from scratch (Hann-windowed ramp filter, edge-replicate padding, approximate
cosine weighting) rather than adapted from VICTRE's own GPL-licensed reconstruction code, which
is neither read nor copied here.

> **Metric note.** Whole-volume correlation is confounded by background behaviour: FBP's ramp
> filter leaks energy into the zero-padded regions of the projections (mean background intensity
> {bgT.loc["FBP","Real MC"]} vs. a ground-truth background of {gt_bg:.4f}), while Aᵀp drives the
> background to near zero "for free". Because background voxels dominate the volume by count,
> whole-volume correlation rewards a clean background over faithful breast reconstruction.
> **Breast-masked correlation (Table 1) is therefore the primary cross-method metric**;
> whole-volume correlation and background energy (Tables 2, 6) are retained as diagnostics.

## Table 1 — Correlation, breast mask (PRIMARY metric)
{corrMaskT.to_markdown()}

## Table 2 — Correlation, whole volume (diagnostic — see metric note)
{corrT.to_markdown()}

## Table 3 — Raw masked PSNR (dB) — scale-dominated in the real regime
{psnrRawT.to_markdown()}

## Table 4 — Scale-matched PSNR (dB) — structural fidelity (scale removed)
{psnrSclT.to_markdown()}

## Table 5 — Inverse-crime gap (ideal − real)
{gapT.to_markdown()}

## Table 6 — Background energy (diagnostic; ground truth = {gt_bg:.4f})
{bgT.to_markdown()}

## Table 7 — Mass SDNR (positive patients; ground truth = {gt_sdnr_txt})
{sdnrT.to_markdown()}

---
### Suggested captions

**Figure 4.** Baseline reconstruction under two regimes (n={N_PAT}), nine methods including FBP.
(a) Correlation (breast mask) vs. SIRT iterations: the inverse-crime regime improves with
iteration, the inverse-crime-free regime degrades. (b) Per-method correlation, ordered by
inverse-crime gap. (c) The gap — in both correlation and scale-matched PSNR — grows monotonically
with a method's reliance on the forward operator, from FBP (a single filtered back-projection,
no data-consistency loop) to SIRT-100 (100 enforcements). (d) Mass SDNR, same ordering; dashed
line marks the ground-truth SDNR.

**Figure 6.** Raw (a) vs. scale-matched (b) masked PSNR. Raw PSNR in the real regime is
scale-dominated; scale-matched PSNR isolates structural fidelity and reproduces the same
ordering as correlation, including FBP's near-zero inverse-crime gap.

### Baseline citations
FBP ramp filter (Kak & Slaney) · SIRT (Gilbert, 1972) · SART (Andersen & Kak, 1984) ·
ASD-POCS (Sidky & Pan, 2008) · LEAP projector (Kim & Champley, LLNL) ·
VICTRE (Badano et al., 2018) · inverse crime (Kaipio & Somersalo, 2005).
"""
save_text(md, "baseline_tables.md")
main.to_csv(os.path.join(LOCAL_DIR, "rows_test_final.csv"), index=False)
_push(os.path.join(LOCAL_DIR, "rows_test_final.csv"))

print("\n" + "="*72 + "\n🎉 DONE — tables + Figure 4 + Figure 6 (9 methods, FBP incl.)\n" + "="*72)
print(f"  Drive: {PAPER_DIR}")
print("\n  Table 1 — correlation (breast mask):"); print(corrMaskT.to_string())
print("\n  Table 5 — inverse-crime gap:"); print(gapT.to_string())
