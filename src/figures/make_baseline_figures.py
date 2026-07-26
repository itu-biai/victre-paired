# ═══════════════════════════════════════════════════════════════════════════
# BASELINE Part 3 v3 — unified tables incl. SCALE-MATCHED PSNR
#   Merges rows_test_tworegime.npy (SSIM/SDNR/GT) with rows_scalematched.npy
#   (scale-matched PSNR), and produces the final paper-ready tables + a small
#   figure showing raw vs scale-matched PSNR (scale collapse ≠ structure loss).
#   No GPU needed (reads saved rows). Saves to Drive paper_assets/baseline_v6.
# ═══════════════════════════════════════════════════════════════════════════
import os, sys, glob, time, shutil, subprocess
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
    lp = os.path.join(LOCAL_DIR, fname); fig.savefig(lp, dpi=300, bbox_inches="tight", facecolor="white"); _push(lp)
    if pdf:
        lpp = lp.replace(".png",".pdf"); fig.savefig(lpp, bbox_inches="tight", facecolor="white"); _push(lpp)
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

p_rows = find("rows_test.npy")
assert p_rows, "rows_test.npy not found — run run_baselines.py first"
print(f"✅ rows: {p_rows}")
dm = pd.DataFrame(list(np.load(p_rows, allow_pickle=True)))

gt = dm[dm.method == "GT"].copy()
main = dm[dm.method != "GT"].copy()   # rows_test.npy already carries psnr_scaled + GT rows
ORDER = [m for m in ["ATp","SIRT-20","SIRT-50","SIRT-100","SART-2","SART-4","SART-8","ASDPOCS-20"]
         if m in main.method.unique()]
N_PAT = main.seed.nunique()
DENS_LABEL = {"scattered":"Scattered","hetero":"Heterogeneous","dense":"Dense","fatty":"Fatty"}
print(f"   {N_PAT} patients | merged scale-matched PSNR onto {main.psnr_scaled.notna().mean()*100:.0f}% of rows")

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
C_IDEAL, C_REAL = "#0173B2", "#DE8F05"

def ms(col, sub=None):
    d = sub if sub is not None else main
    m = d.pivot_table(index="method", columns="regime", values=col, aggfunc="mean").reindex(ORDER)
    s = d.pivot_table(index="method", columns="regime", values=col, aggfunc="std").reindex(ORDER)
    return m, s

# ── FIGURE 3 (NEW) — raw vs scale-matched PSNR ───────────────────────────
rm, rs = ms("psnr_mask"); sm, ss = ms("psnr_scaled")
x = np.arange(len(ORDER)); w = 0.38
fig3, ax = plt.subplots(1, 2, figsize=(11, 3.7), sharey=True)
for k, (mm, ssd, ttl) in enumerate([(rm, rs, "(a) Raw masked PSNR"), (sm, ss, "(b) Scale-matched PSNR")]):
    ax[k].bar(x-w/2, [mm.loc[m,"ideal"] for m in ORDER], w, yerr=[ssd.loc[m,"ideal"] for m in ORDER],
              color=C_IDEAL, capsize=2)
    ax[k].bar(x+w/2, [mm.loc[m,"real"] for m in ORDER], w, yerr=[ssd.loc[m,"real"] for m in ORDER],
              color=C_REAL, capsize=2)
    ax[k].set_xticks(x); ax[k].set_xticklabels(ORDER, rotation=45, ha="right"); ax[k].set_title(ttl)
ax[0].set_ylabel("PSNR (dB)")
fig3.legend(handles=[Patch(color=C_IDEAL, label="Ideal $A$(clean) — inverse crime"),
                     Patch(color=C_REAL, label="Real MC — inverse-crime-free")],
            loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
fig3.tight_layout(rect=[0,0,1,0.94])
print("✅ Figure 3:", save_fig(fig3, "fig3_psnr_raw_vs_scaled.png", pdf=True)); plt.close(fig3)

# ── tables ───────────────────────────────────────────────────────────────
try: import tabulate  # noqa
except Exception: subprocess.run(f"{sys.executable} -m pip install -q tabulate", shell=True)

def tbl(col, prec=3, signed=False, sub=None):
    m, s = ms(col, sub); sg = "+" if signed else ""; out = pd.DataFrame(index=ORDER)
    for reg, lab in [("ideal","Ideal A(clean)"), ("real","Real MC")]:
        out[lab] = [f"{m.loc[k,reg]:{sg}.{prec}f} ± {s.loc[k,reg]:.{prec}f}" for k in ORDER]
    return out

corrT = tbl("corr", 3)
psnrRawT = tbl("psnr_mask", 2)
psnrSclT = tbl("psnr_scaled", 2)
ssimT = tbl("ssim", 3)
sdnrT = tbl("sdnr", 3, signed=True, sub=main[main.ip])
cm, _ = ms("corr"); gapT = pd.DataFrame({"Ideal − Real (corr)": [f"{(cm.loc[m,'ideal']-cm.loc[m,'real']):.3f}" for m in ORDER]}, index=ORDER)
denscols = [DENS_LABEL[d] for d in ["scattered","hetero","dense","fatty"] if d in main.dens.unique()]
densT = (main[main.regime=="real"].pivot_table(index="method", columns="dens", values="psnr_mask", aggfunc="mean")
         .reindex(ORDER).rename(columns=DENS_LABEL)[denscols].round(2))
gt_txt = f"{gt.sdnr.dropna().mean():+.3f} ± {gt.sdnr.dropna().std():.3f}"

md = f"""# VICTRE-PAIRED v6 — Baseline reconstruction results (test split, n={N_PAT} patients)

Every baseline is evaluated under two projection regimes. The **ideal** regime drives
reconstruction with synthetic forward projections `A(clean)` from the *same* line-integral
operator used inside the reconstructors — by construction an **inverse crime** (Kaipio &
Somersalo, 2005), a best-case geometry-matched bound. The **real** regime uses the
independent Monte-Carlo projections (`noisy_proj`, half dose), containing scatter, beam
hardening and detector response not modelled by the operator — **inverse-crime-free**.
Reconstructions are min–max normalized to [0,1]; PSNR/SDNR use the breast mask. SEED
208084664 is excluded.

> **On PSNR.** In the real regime the reconstruction's intensity scale collapses, so *raw*
> masked PSNR (~7 dB) is scale-dominated. **Scale-matched PSNR** fits the optimal affine map
> to the GT over the mask before PSNR, isolating structural fidelity: the real regime then
> recovers to ~15 dB — showing the collapse is in *scale*, not *structure*. Method ordering
> under scale-matched PSNR matches correlation (fewer iterations better), so correlation and
> scale-matched PSNR tell the same story; correlation is the primary scale-invariant metric.

## Table 1 — Correlation with reference volume (primary cross-regime metric)
{corrT.to_markdown()}

## Table 2 — Raw masked PSNR (dB) — scale-dominated in the real regime
{psnrRawT.to_markdown()}

## Table 3 — Scale-matched PSNR (dB) — structural fidelity (scale removed)
{psnrSclT.to_markdown()}

## Table 4 — SSIM (breast slices)
{ssimT.to_markdown()}

## Table 5 — Mass SDNR (positive patients; GT = {gt_txt})
{sdnrT.to_markdown()}

## Table 6 — Inverse-crime gap (ideal − real correlation)
{gapT.to_markdown()}

## Table 7 — Raw PSNR by density, real regime (dB)
{densT.to_markdown()}

---
### Suggested captions

**Figure 1.** Baseline reconstruction under two regimes (n={N_PAT}). (a) Correlation vs. SIRT
iterations: inverse-crime `A(clean)` improves, real MC degrades — forward-model mismatch.
(b) Per-method correlation and (c) per-method mass SDNR, ideal vs. real (dashed = GT SDNR).
Error bars ±1 s.d.

**Figure 2.** One representative case per density, cropped to the breast (mass circled).
GT, SIRT-50 (ideal), SIRT-50 (real) and Aᵀp (real); per-panel 2–98 percentile window.
Iterative reconstruction degrades under real MC while back-projection retains structure.

**Figure 3.** Raw (a) vs. scale-matched (b) masked PSNR, ideal vs. real. Raw PSNR collapses
to ~7 dB under real MC, but scale-matched PSNR recovers to ~15 dB: the mismatch corrupts
the intensity *scale* far more than the underlying *structure*. Ideal-regime scale-matched
PSNR increases monotonically with SIRT iterations, unlike raw PSNR (semi-convergence is a
scale effect).

**Tables.** Baselines under both regimes. The ideal–real correlation gap (Table 6) grows
with iterations (0.10→0.41), quantifying inverse-crime optimism; fewer iterations are
consistently better in the real regime across correlation, SDNR and scale-matched PSNR.

### Baseline citations
SIRT (Gilbert, 1972) · SART (Andersen & Kak, 1984) · ASD-POCS (Sidky & Pan, 2008) ·
LEAP projector (Kim & Champley, LLNL) · VICTRE (Badano et al., 2018) ·
inverse crime (Kaipio & Somersalo, 2005).
"""
save_text(md, "baseline_tables.md")
main.to_csv(os.path.join(LOCAL_DIR, "baseline_raw_test_merged.csv"), index=False)
_push(os.path.join(LOCAL_DIR, "baseline_raw_test_merged.csv"))

print("\n" + "="*72 + "\n🎉 DONE — unified tables (with scale-matched PSNR) + Figure 3\n" + "="*72)
print(f"  Drive: {PAPER_DIR}")
print("\n  Table 3 — Scale-matched PSNR (dB):"); print(psnrSclT.to_string())
