# ═══════════════════════════════════════════════════════════════════════════
# DATASET CHARACTERIZATION FIGURES v2 — English, publication quality
#   Fixes vs v1:
#     • Curve (a): full 25-angle tissue-level profile (was only 3 points).
#     • Montage: crop projection panels to breast (shared bbox, parallax kept)
#       and reconstruction to its own bbox → breast fills panels, dose noise
#       visible. Mass circle shifted to the crop.
#     • Caption file flush bug fixed (close before copying to Drive).
#   Reads a representative sample (DEEP_N chunks, heavy). ~6 min. Saves to
#   Drive paper_assets/dataset (+ local), with retry.
# ═══════════════════════════════════════════════════════════════════════════
import os, sys, glob, time, gc, shutil
import numpy as np

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
except Exception as e:
    print("drive.mount skipped:", e)

V6        = "/content/drive/MyDrive/New_DBT/VICTRE-PAIRED-v6"
PAPER_DIR = "/content/drive/MyDrive/New_DBT/paper_assets/dataset"
LOCAL_DIR = "/content/paper_dataset"
for d in (LOCAL_DIR, PAPER_DIR):
    try: os.makedirs(d, exist_ok=True)
    except Exception as e: print(f"  mkdir ({d}):", e)

DEEP_N, MAX_FILL = 40, 40
BROKEN, NA = {208084664}, 25
WANT = ["scattered", "hetero", "dense", "fatty"]
DENS_LABEL = {"scattered":"Scattered", "hetero":"Heterogeneous", "dense":"Dense", "fatty":"Fatty"}

def fmt(s):
    s = int(s); h, s = divmod(s, 3600); m, s = divmod(s, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"
def safe(f, t=3):
    for i in range(t):
        try: return np.load(f)
        except Exception:
            if i == t-1: return None
            time.sleep(2); gc.collect()
def gl(pat, t=5):
    for i in range(t):
        r = sorted(glob.glob(pat))
        if r: return r
        time.sleep(3)
    return []
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
def save_text(text, fname):                    # write + CLOSE, then copy (v1 bug fix)
    lp = os.path.join(LOCAL_DIR, fname)
    with open(lp, "w") as fh: fh.write(text)
    _push(lp); return lp

def sdnr(vol, z, h, w, ri=4, ro=12):
    z, h, w = int(round(z)), int(round(h)), int(round(w))
    if not (0 <= z < vol.shape[0] and 0 <= h < vol.shape[1] and 0 <= w < vol.shape[2]): return np.nan
    c = vol[max(0,z-1):z+2, max(0,h-ri):h+ri+1, max(0,w-ri):w+ri+1]
    R = vol[z, max(0,h-ro):h+ro+1, max(0,w-ro):w+ro+1]
    return (c.mean()-R.mean())/R.std() if R.std() > 1e-8 else np.nan
def prof90(x):                                  # per-angle 90th-pct tissue level (25 values)
    return [float(np.percentile(x[a][x[a] > 0], 90)) if (x[a] > 0).any() else np.nan for a in range(NA)]

# ── 1) Sample scan ───────────────────────────────────────────────────────
files = [f for sp in ("train","val","test") for f in gl(f"{V6}/{sp}/*.npz")]
np.random.default_rng(0).shuffle(files)
print(f"total chunks: {len(files)} | sampling {DEEP_N} for curves", flush=True)

u_all, sf_all, sh_all, sq_all, ms_all, cs_all = [], [], [], [], [], []
picks = {}; t0 = time.time()

def ingest(d):
    cp = d["clean_proj"]; cl = d["clean"]; lc = d["lesion_coords"]; nl = d["lesion_count"]
    sf_all.extend(map(float, d["sigma_full"])); sh_all.extend(map(float, d["sigma"]))
    sq_all.extend(map(float, d["sigma_quarter"]))
    for b in range(len(d["seed"])):
        if int(d["seed"][b]) in BROKEN: continue
        x = cp[b]; lv = [np.percentile(x[a][x[a] > 0], 90) for a in range(NA) if (x[a] > 0).any()]
        if lv: u_all.append(max(lv)/min(lv))
        for z, h, w, t in lc[b][:nl[b]]:
            s = sdnr(cl[b], z, h, w)
            if not np.isnan(s): (ms_all if t >= 4 else cs_all).append(s)
        dn = str(d["density"][b])
        if dn in WANT and dn not in picks and bool(d["is_pos"][b]):
            m = lc[b][:nl[b]]; m = m[m[:,3] >= 4]
            if len(m):
                z = int(round(m[0,0]))
                if 0 <= z < cl.shape[1]:
                    picks[dn] = dict(p0=x[0].copy(), p12=x[12].copy(), p24=x[24].copy(),
                                     prof=prof90(x), rec=cl[b].copy(), z=z,
                                     lh=float(m[0,1]), lw=float(m[0,2]),
                                     nf=d["noisy_proj_full"][b][12].copy(),
                                     nq=d["noisy_proj_quarter"][b][12].copy(), seed=int(d["seed"][b]))

for i, f in enumerate(files[:DEEP_N]):
    d = safe(f)
    if d is None: continue
    ingest(d); del d; gc.collect()
    k = i + 1
    if k % 5 == 0 or k == DEEP_N:
        e = time.time()-t0
        print(f"  curves {k}/{DEEP_N} | {fmt(e)} | ETA ~{fmt(e/k*(DEEP_N-k))} | montage {sorted(picks)}", flush=True)

missing = [dn for dn in WANT if dn not in picks]
if missing:
    print(f"  filling montage densities: {missing}", flush=True)
    for f in files[DEEP_N:DEEP_N+MAX_FILL]:
        if not missing: break
        d = safe(f)
        if d is None: continue
        if {str(x) for x in d["density"]} & set(missing):
            ingest(d); missing = [dn for dn in WANT if dn not in picks]
        del d; gc.collect()
order_p = [dn for dn in WANT if dn in picks]
print(f"  montage complete: {order_p} | curve n(patients)≈{len(u_all)}", flush=True)

# ── 2) Style ─────────────────────────────────────────────────────────────
import matplotlib; matplotlib.use("Agg")
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import Circle
mpl.rcParams.update({
    "savefig.dpi": 300, "figure.dpi": 120, "font.family": "DejaVu Sans",
    "font.size": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white", "axes.axisbelow": True,
})
u = np.array(u_all); sf = np.array(sf_all); sh = np.array(sh_all); sq = np.array(sq_all)
ms = np.array(ms_all); cs = np.array(cs_all)

def bbox(im, thr=0.05, mg=10):
    ys, xs = np.where(im > thr)
    if len(ys) == 0: return 0, im.shape[0], 0, im.shape[1]
    return (max(0,ys.min()-mg), min(im.shape[0],ys.max()+mg),
            max(0,xs.min()-mg), min(im.shape[1],xs.max()+mg))

# ── FIGURE D1 — montage (cropped) ────────────────────────────────────────
cols = ["Projection −25°", "Projection 0°", "Projection +25°",
        "Reconstruction (GT)", "Full-dose proj (0°)", "Quarter-dose proj (0°)"]
fig1, ax = plt.subplots(len(order_p), 6, figsize=(3.6*6, 3.6*len(order_p)))
if len(order_p) == 1: ax = ax.reshape(1, 6)
for i, dn in enumerate(order_p):
    P = picks[dn]; z = P["z"]
    py0, py1, px0, px1 = bbox(P["p12"])                       # shared projection crop (parallax kept)
    ry0, ry1, rx0, rx1 = bbox(P["rec"][z])                    # reconstruction crop
    proj_panels = [P["p0"], P["p12"], P["p24"], None, P["nf"], P["nq"]]
    for j in range(6):
        if j == 3:
            sl = P["rec"][z][ry0:ry1, rx0:rx1]
            vmn, vmx = np.percentile(sl, [1, 99])
            ax[i, j].imshow(sl, cmap="gray", vmin=vmn, vmax=vmx)
            ax[i, j].add_patch(Circle((P["lw"]-rx0, P["lh"]-ry0), radius=10, fill=False,
                                      edgecolor="#00E5FF", lw=1.4))
        else:
            sl = proj_panels[j][py0:py1, px0:px1]
            ax[i, j].imshow(sl, cmap="gray", vmin=0, vmax=1)   # shared window → dose noise comparable
        head = cols[j] if i == 0 else ""
        sub  = (f"{DENS_LABEL[dn]}\nSEED {P['seed']}" if j == 0 else (f"z={z}" if j == 3 else ""))
        ax[i, j].set_title((head + ("\n" if head and sub else "") + sub).strip(), fontsize=9.5)
        ax[i, j].axis("off")
fig1.suptitle("VICTRE-PAIRED v6 — paired projections, reconstruction and multi-dose "
              "(one positive case per density)", y=1.004, fontsize=12)
fig1.tight_layout()
print("✅ Figure D1:", save_fig(fig1, "figD1_dataset_montage.png")); plt.close(fig1)

# ── FIGURE D2 — curves ───────────────────────────────────────────────────
fig2, ax = plt.subplots(1, 4, figsize=(19, 4.3))
ang = np.linspace(-25, 25, NA)
for dn in order_p:
    ax[0].plot(ang, picks[dn]["prof"], "-", lw=1.8, label=DENS_LABEL[dn])
ax[0].set_xlabel("Projection angle (deg)"); ax[0].set_ylabel("Tissue level (90th pct)")
ax[0].set_title("(a) Per-angle tissue level"); ax[0].legend(frameon=False, fontsize=8.5)
ax[1].hist(u, bins=30, color="#4C78A8", edgecolor="white", linewidth=0.4)
ax[1].axvline(1.10, color="#D1495B", ls="--", lw=1.5, label="physics 1/cos25° = 1.10")
ax[1].set_xlabel("Uniformity ratio (max/min)"); ax[1].set_ylabel("Patients")
ax[1].set_title(f"(b) Projection uniformity (n={len(u)})"); ax[1].legend(frameon=False, fontsize=8.5)
ax[2].hist([sf, sh, sq], bins=25, label=["Full", "Half", "Quarter"], color=["#0173B2", "#029E73", "#DE8F05"])
ax[2].set_xlabel("Measured noise σ"); ax[2].set_ylabel("Patients")
ax[2].set_title("(c) Noise by dose level"); ax[2].legend(frameon=False, fontsize=8.5)
ax[3].hist([ms, cs], bins=30, label=[f"Mass (n={len(ms)})", f"Calcification (n={len(cs)})"],
           color=["#0173B2", "#DE8F05"])
ax[3].axvline(0, color="k", ls=":", lw=1.2); ax[3].set_xlabel("SDNR"); ax[3].set_ylabel("Lesions")
ax[3].set_title("(d) Lesion SDNR"); ax[3].legend(frameon=False, fontsize=8.5)
fig2.tight_layout()
print("✅ Figure D2:", save_fig(fig2, "figD2_dataset_curves.png", pdf=True)); plt.close(fig2)

# ── captions ─────────────────────────────────────────────────────────────
cap = f"""### Dataset-characterization figures — suggested captions

**Figure D1.** One signal-present case per breast density (panels cropped to the breast).
Columns: three of the 25 limited-angle Monte-Carlo projections (−25°, 0°, +25°; the breast
shifts with view angle, illustrating the parallax of the ±25° sweep); the paired FBP
reconstruction (central mass-bearing slice, mass circled in cyan); and the full- and
quarter-dose noisy projections at 0°, illustrating the multiple dose levels provided.

**Figure D2.** Physical fidelity of VICTRE-PAIRED v6 (representative subset, n≈{len(u)}
patients). (a) Per-angle 90th-percentile tissue level across the ±25° sweep (tissue level
rises toward oblique views, consistent with the 1/cos path-length increase). (b) Projection
uniformity (max/min of per-angle tissue level); dashed line marks the expected
1/cos25°=1.10. (c) Measured noise σ at the three dose levels, showing clean separation and
a monotonic increase. (d) Lesion SDNR: masses (tip 4–7) are measurable while
microcalcifications (tip 0–3) sit at the noise floor after 4× spatial downsampling, so the
release supports mass tasks. Exact dataset-wide statistics are in the Technical Validation
(v6_validation.json).
"""
save_text(cap, "dataset_figure_captions.md")

print("\n" + "="*72 + "\n🎉 DONE — English dataset figures ready\n" + "="*72)
print(f"  Drive : {PAPER_DIR}\n  Local : {LOCAL_DIR}")
for f in ["figD1_dataset_montage.png", "figD2_dataset_curves.png/.pdf", "dataset_figure_captions.md"]:
    print("    •", f)
print(f"\n  sample stats: uniformity {u.mean():.3f} (physics 1.10) | "
      f"σ {sf.mean():.4f}/{sh.mean():.4f}/{sq.mean():.4f} | "
      f"mass SDNR {ms.mean():+.3f} (n={len(ms)}) | calc SDNR {cs.mean():+.3f} (n={len(cs)})")
