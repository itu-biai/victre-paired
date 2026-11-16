"""Generates 02_figures_tables_colab.ipynb. The figure and table plan of the Data Descriptor: 7 main figures + 4 main tables + supplementary material."""
import json, sys
from pathlib import Path

OUT = Path(sys.argv[1])
cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md(r"""
# VICTRE-Paired v11 — figures, tables, `numbers.tex` (Colab, CPU)

The figure and table plan of the Data Descriptor: **7 main figures, 4 main tables**,
training and details in the supplementary material. No GPU needed. **A → B** in every session; the cells after that are
independent (~10 min). An item whose source is missing is skipped and the reason is printed; the notebook can run
on intermediate results for a preview.

| item | content | source |
|---|---|---|
| Fig 1 | pipeline (TikZ, `paper/`): this notebook produces only the **images** → `FIGURES/fig1_thumbs/` | `extra/panel` |
| Fig 2 | geometry: side and top view, forward-model check, residual per view | config, `extra_viewres`, `extra/panel` |
| Fig 3 | example records: one row per density | `extra/panel` |
| Fig 4 | dataset statistics | `registry`, DIDSR `.loc` |
| Fig 5 | dose fidelity: noise model, NPS, density × dose PSNR, mammogram | `extra_noisebins`, `extra_nps`, `baseline`, `extra/panel` |
| Fig 6 | methods × dose, error maps, PSNR and rank under two projection sources | `extra_metrics`, `extra/panel` |
| Fig 7 | lesions: mass and calcification slices, depth profile, before/after DELTA (tissue offset), SDNR | `extra_sdnr`, `extra/panel`, `delta/zshift` (06), `delta_patch_s*.jsonl` in the dataset (07) |
| Tab 1 | record schema | one test file |
| Tab 2 | split × density composition | `registry` |
| Tab 3 | classical methods × dose (MC) | `extra_metrics`, `extra_sdnr` |
| Tab 4 | projection source and second reference (25 %) | `extra_metrics`, `training_metrics` |
| Supp. | S1 Tab 3 by density · S2 geometry constants · S3 training · S4 detection AUC · S5 noise (all files) · Fig S1 training · Fig S2 calcification gallery · Fig S3 `victre_fbp`–`sart_full` scatter | |

**Rules (plan):** every number comes from `numbers.tex`; metrics on the test split, inside the breast mask,
mean ± s.d. over phantoms; one display window per row/figure (the inner breast region,
the same across doses and methods); a 10 mm scale bar; **density = colour (Okabe–Ito) + marker,
dose = lightness of a single hue + marker, method = line style**; a result appears in a figure or in a table, not both.

**Scope:** the ranking under the two projection sources and the numbers against `victre_fbp` are
**reported, not interpreted**; their cause is the subject of the companion study.
""")

md("## A · Drive")
code(r"""
from google.colab import drive
drive.mount('/content/drive')
""")

md("## B · Settings — paths, style, data, helpers")
code(r"""
import os, re, json, math, csv, io, tarfile, urllib.request, ast
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import ndimage
from scipy.stats import rankdata

GEN  = Path("/content/drive/MyDrive/victre_paired/code/dataset_generate.py")   # the constants are read from here
DATA = Path("/content/drive/MyDrive/victre_paired/data")
RES  = Path("/content/drive/MyDrive/victre_paired/analysis")
FIG, TAB, THUMB = RES/"FIGURES", RES/"TABLES", RES/"FIGURES"/"fig1_thumbs"
for d in (FIG, TAB, THUMB): d.mkdir(parents=True, exist_ok=True)

FMT, DPI = ("pdf", "png"), 300
W1, W2 = 3.50, 7.20                         # 89 mm and 183 mm [inch]
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.edgecolor": "#555555", "xtick.color": "#555555", "ytick.color": "#555555",
    "axes.labelcolor": "#222222", "text.color": "#222222",
    "lines.linewidth": 1.3, "figure.dpi": 110, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
})
# Colour scheme (the plan's "fixed meaning" rule; 8 Okabe-Ito colours cannot carry 12+ meanings):
#   density = Okabe-Ito colour + marker   (colour-blindness test: CVD dE >= 9)
#   dose    = lightness ramp of one hue + marker (dark = full dose)
#   method  = line style + marker, ink colour
DENS  = ["dense", "hetero", "scattered", "fatty"]
DNAME = {"dense": "Dense", "hetero": "Heterogeneous", "scattered": "Scattered", "fatty": "Fatty"}
DSHORT = {"dense": "Dense", "hetero": "Het.", "scattered": "Scat.", "fatty": "Fatty"}
DCOL  = dict(zip(DENS, ["#0072B2", "#D55E00", "#009E73", "#E69F00"]))
DMARK = dict(zip(DENS, ["o", "s", "^", "D"]))
DOSES = ["50", "25", "10"]
DOSE_ALL = ["100", "50", "25", "10"]
DOSECOL = {"100": "#1B1B1B", "50": "#2F5F98", "25": "#6F9BD1", "10": "#B9CFEA"}
DOSEMARK = {"100": "o", "50": "s", "25": "^", "10": "v"}
DLAB = {"100": "full", "50": "50 %", "25": "25 %", "10": "10 %"}
DOSEW = {"100": "Full", "50": "Fifty", "25": "TwentyFive", "10": "Ten"}
INK, MUTED, GRID = "#222222", "#777777", "#DDDDDD"
METH = ["FBP", "SART", "OS-SART", "SART-TV"]
MSTY = {"FBP": dict(ls="--", marker="s"), "SART": dict(ls=":", marker="^"),
        "OS-SART": dict(ls="-", marker="o"), "SART-TV": dict(ls="-.", marker="D"),
        "U-Net": dict(ls="-", marker="*")}
MKEY = {m: m.replace("-", "") for m in METH}            # panel keys
MW = {"FBP": "FBP", "SART": "SART", "OS-SART": "OSSART", "SART-TV": "SARTTV", "U-Net": "UNet"}
SRCLAB = {"mc": "Monte Carlo", "syn": "synthetic $A x_\\mathrm{ref}$"}
LESCOL = {1: "#56B4E9", 2: "#D55E00", 3: "#F0E442"}      # lesion_roi: 1 calcification, 2 mass, 3 control

MADE, SKIPPED, NUM = [], [], {}
def save(fig, name, sub=FIG):
    for ext in FMT: fig.savefig(sub/f"{name}.{ext}", dpi=DPI)
    MADE.append(name); print(f"   -> {name}.{'/'.join(FMT)}")
def panel(ax, letter): ax.set_title(f"({letter})", loc="left", fontweight="bold", pad=3)
def tag(ax, letter, x=-0.12, y=1.32):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontweight="bold", fontsize=7, ha="left", va="bottom")
def need(name, *objs):
    if all(o is not None and (not hasattr(o, "__len__") or len(o)) for o in objs): return True
    SKIPPED.append(name); print(f"   [skipped] {name}: source missing"); return False
def csv_or_none(name, **kw):
    p = RES/name
    return pd.read_csv(p, **kw) if p.exists() else None
def npz_dir(d):
    return {dn: np.load(d/f"{dn}.npz", allow_pickle=False) for dn in DENS if (d/f"{dn}.npz").exists()}

STR = dict(dtype={"patient": str, "dose": str, "train_dose": str})
K, B, SD = csv_or_none("registry.csv", **STR), csv_or_none("baseline.csv", **STR), csv_or_none("sdnr.csv", **STR)
Z, DF, NZ = csv_or_none("zalign.csv", **STR), csv_or_none("datafit.csv", **STR), csv_or_none("noise.csv", **STR)
EM, ES = csv_or_none("extra_metrics.csv", **STR), csv_or_none("extra_sdnr.csv", **STR)
VR, NB, NP = csv_or_none("extra_viewres.csv", **STR), csv_or_none("extra_noisebins.csv", **STR), csv_or_none("extra_nps.csv", **STR)
TM, TS, TC = csv_or_none("training_metrics.csv", **STR), csv_or_none("training_sdnr.csv", **STR), csv_or_none("training_curve.csv")
for df in (TM, TS):
    if df is not None: df["train_dose"] = df["train_dose"].fillna("").astype(str)
PAN01, PAN = npz_dir(RES/"panel"), npz_dir(RES/"extra"/"panel")
EX = np.load(RES/"training"/"example.npz") if (RES/"training"/"example.npz").exists() else None
SETTINGS = json.loads((RES/"extra"/"settings.json").read_text()) if (RES/"extra"/"settings.json").exists() else None
# 06 (on v11): tissue z offset, DELTA scan, mask edge. 05: scale, noise regeneration.
def sub_csv(d, name, **kw):
    p = RES/d/name
    return pd.read_csv(p, **kw) if p.exists() else None
ZS, ZSS = sub_csv("delta", "zshift.csv", **STR), sub_csv("delta", "zshift_summary.csv")
DSC, MED = sub_csv("delta", "scan_summary.csv"), sub_csv("delta", "mask_edge.csv", **STR)
SCT, NRG = sub_csv("diagnostics", "scale_test.csv", **STR), sub_csv("diagnostics", "noise_regen.csv", **STR)
ZST = sub_csv("diagnostics", "zalign_stacked.csv")
# log of 07 (DELTA patch): tissue offset before/after the patch, for every patched file. Inside the dataset folder.
_yl = [json.loads(l) for f in sorted(DATA.glob("delta_patch_s*.jsonl")) for l in f.read_text().splitlines() if l.strip()]
PATCH = pd.DataFrame([r for r in _yl if r.get("status") == "patched"]) if _yl else None
if PATCH is not None and len(PATCH): PATCH = PATCH.drop_duplicates("patient", keep="last")
else: PATCH = None
PATCH_DELTA_OLD = {"dense": 19.765, "hetero": 20.242, "scattered": 20.091}   # DELTA_OLD of 07
VAL = (DATA/"validation.md").read_text() if (DATA/"validation.md").exists() else None
CFGJ = json.loads((DATA/"config.json").read_text()) if (DATA/"config.json").exists() else {}
SIDc, SDDc = float(CFGJ.get("sid_mm", 600.0)), float(CFGJ.get("sdd_mm", 650.0))
PIX = float(CFGJ.get("native_pixel_mm", 0.085)) * int(CFGJ.get("downsample", 4))   # 0.34 mm
NV, ANG = int(CFGJ.get("num_views", 25)), float(CFGJ.get("half_angle_deg", 25.0))
DETR, DETC = int(CFGJ.get("det_rows", 752)), int(CFGJ.get("det_cols", 384))
VOLR, VOLC = int(CFGJ.get("vol_rows", 408)), int(CFGJ.get("vol_cols", 336))
# DELTA, slice counts and volume-box constants from the generator: the figures and tables must use the same
# values that produced the data (typed by hand they would silently drift when the code changes).
_gen = GEN.read_text()
def _const(name): return ast.literal_eval(re.search(rf"^{name} = (.+?)(?:\s+#.*)?$", _gen, re.M).group(1))
DELTA, SLC, NATR, OFFX_C = _const("DELTA"), _const("DENSITY_SLICES"), _const("NATIVE_ROWS"), _const("OFFX_C")
assert all(v == 19.5 for v in DELTA.values()), f"dataset_generate.py is old (before the DELTA patch): {DELTA}"
DELTA_PUB = 19.5          # VICTRE: d_objbottom_det 20.0 mm (FBP_DBTrecon.c:869) - half of the 0.5 mm cover
# Before the fatty correction: pilot run (30 fatty phantoms, 60 masses) with DELTA[fatty] = 20.807. Measured
# during generation; not in the release, carried here as a constant.
PILOT_FATTY = dict(delta=20.807, gt=(-1.20, 0.12, 60), vfbp=(+0.10, 0.17, 12))
def volume_box(dn, S):
    hx, hy = VOLR * PIX / 2, VOLC * PIX / 2
    offx = (VOLR * PIX - NATR[dn] * float(CFGJ.get("native_pixel_mm", 0.085))) / 2.0 + OFFX_C
    cw = -DETC * PIX / 2; offy = cw + VOLC * PIX / 2
    offz = -(SDDc - SIDc) + S / 2.0 + DELTA[dn]
    return offx - hx, offx + hx, offy - hy, offy + hy, offz - S / 2, offz + S / 2

print("DELTA", DELTA, "| slices", SLC)
for n, o in (("registry", K), ("baseline", B), ("sdnr", SD), ("zalign", Z), ("datafit", DF), ("noise", NZ),
             ("extra_metrics", EM), ("extra_sdnr", ES), ("extra_viewres", VR), ("extra_noisebins", NB), ("extra_nps", NP),
             ("training_metrics", TM), ("training_curve", TC), ("validation.md", VAL), ("config.json", CFGJ or None),
             ("delta/zshift", ZS), ("delta/scan_summary", DSC), ("delta/mask_edge", MED), ("diagnostics/scale_test", SCT),
             ("diagnostics/noise_regen", NRG), ("delta-patch log", PATCH)):
    print(f"{n:<15} {'MISSING' if o is None else (str(len(o)) + ' rows' if hasattr(o, 'shape') else 'present')}")
print("panel 01:", sorted(PAN01), "| panel 03:", sorted(PAN), "| training example:", EX is not None, "| settings:", SETTINGS is not None)

# ------------------------------------------------------------------ helpers
def tex_int(n): return f"{int(n):,}".replace(",", "{,}")
def tt(s): return r"\texttt{" + s.replace("_", r"\_") + "}"
def pm(m, s, n=2): return f"{m:.{n}f} $\\pm$ {s:.{n}f}"
def cap(s): return s[0].upper() + s[1:]

def write_table(name, source, caption, label, colspec, head, body, resize=False, note=None):
    L = ["% GENERATED AUTOMATICALLY -- do not edit.", f"% source: {source}",
         "% produced by: notebooks/02_figures_tables_colab.ipynb"]
    if note: L.append(f"% not: {note}")
    L += [r"\begin{table}[!tp]", r"\centering", r"\small", r"\caption{" + caption + "}", r"\label{" + label + "}"]
    if resize: L.append(r"\resizebox{\textwidth}{!}{%")
    L += [r"\begin{tabular}{" + colspec + "}", r"\toprule"] + head + [r"\midrule"]
    for row in body:
        L.append(r"\midrule" if row == "MIDRULE" else " & ".join(row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}" + ("}" if resize else ""), r"\end{table}"]
    (TAB/f"{name}.tex").write_text("\n".join(L) + "\n")
    MADE.append(name); print(f"   -> TABLES/{name}.tex")

def auc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0: return np.nan
    r = rankdata(np.concatenate([a, b]))
    return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))

def auc_ci(df, n=1000, seed=0):
    # Mass (kind 1) vs control (kind -1); bootstrap over PHANTOMS (locations within a phantom are dependent).
    pm_ = [g.sdnr.values for _, g in df[df.kind == 1].groupby("patient")]
    pc_ = [g.sdnr.values for _, g in df[df.kind == -1].groupby("patient")]
    if not pm_ or not pc_: return np.nan, np.nan, np.nan, 0, 0
    a0, b0 = np.concatenate(pm_), np.concatenate(pc_)
    rng, vals = np.random.default_rng(seed), []
    for _ in range(n):
        a = np.concatenate([pm_[i] for i in rng.integers(0, len(pm_), len(pm_))])
        b = np.concatenate([pc_[i] for i in rng.integers(0, len(pc_), len(pc_))])
        vals.append(auc(a, b))
    return auc(a0, b0), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(a0), len(b0)

def recessive(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)

import matplotlib.patheffects as pe
OUTLINE = [pe.withStroke(linewidth=1.6, foreground="black")]
def noaxis(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_visible(False)
def scalebar(ax, px_mm, h, w, mm=10, color="white"):
    # Top right corner; the length is in the object plane (px_mm is corrected by the caller).
    L = mm / px_mm; x1 = w - 0.05 * w; y = 0.08 * h
    ax.plot([x1 - L, x1], [y, y], color=color, lw=1.8, solid_capstyle="butt", path_effects=OUTLINE)
    ax.annotate(f"{mm} mm", (x1, y), xytext=(0, -2.5), textcoords="offset points", color=color, ha="right", va="top",
                fontsize=6, path_effects=OUTLINE)            # fixed point spacing: does not overlap the bar in a small panel

# Windows: from the INNER breast region (eroded by 10 mm), fixed along a row/figure.
def win_recon(img, mask, lo=2, hi=90):
    er = ndimage.binary_erosion(mask, iterations=int(round(10 / PIX)))
    return np.percentile(img[er if er.sum() > 200 else mask], [lo, hi])
def win_proj(img, valid, px_mm, lo=1, hi=99):
    fp = (img > 0) & valid
    er = ndimage.binary_erosion(fp, iterations=int(round(10 / px_mm)))
    return np.percentile(img[er if er.sum() > 200 else fp], [lo, hi])
def bbox(mask, pad=6):
    r, c = np.nonzero(mask)
    return max(r.min() - pad, 0), r.max() + pad, max(c.min() - pad, 0), c.max() + pad
PIXP = PIX * SIDc / SDDc                 # projection pixel in the object plane
PIXM = PIX / 2 * SIDc / SDDc             # mammogram binned 2x
PICK = lambda d: next((dn for dn in ("scattered", "hetero", "dense", "fatty") if dn in d), None)
print("ready")
""")

md(r"""
## C · Main tables
""")
code(r"""
# ------------------------------------------------------------ Table 1: record schema
DESC = {
    "proj_full": ("line integral", "DBT projections at the trial's dose"),
    "proj_50": ("line integral", "projections at 50\\,\\% of that dose"), "proj_25": ("line integral", "at 25\\,\\%"),
    "proj_10": ("line integral", "at 10\\,\\%"), "proj_valid": ("--", "valid rays; collimator edges excluded"),
    "sart_full": ("mm$^{-1}$", "reference $x_\\mathrm{ref}$: OS-SART of \\texttt{proj\\_full}"),
    "breast_mask": ("--", "projection hull and reference threshold"),
    "lesion_roi": ("--", "approximate labels: 1 calc. cluster, 2 mass, 3 control"),
    "victre_fbp": ("trial units", "the trial's reconstruction on the same grid, second reference; value$/65535\\times$scale"),
    "victre_fbp_present": ("--", "read before \\texttt{victre\\_fbp}"),
    "victre_fbp_scale": ("--", "maximum of the source volume (trial units), $\\texttt{victre\\_fbp}/65535\\times$scale"),
    "victre_fbp_saturated": ("--", "fraction of source voxels at the uint16 ceiling"),
    "dm_full": ("line integral", "mammogram at the trial's dose"), "dm_50": ("line integral", "mammogram at 50\\,\\%"),
    "dm_25": ("line integral", "at 25\\,\\%"), "dm_10": ("line integral", "at 10\\,\\%"),
    "dm_valid": ("--", "valid pixels of the mammogram"),
    "lesions": ("voxel; code", "slice, row, column, lesion number, type (1 mass, 0 calc. cluster, $-1$ control)"),
    "geometry_vectors": ("mm", "ASTRA \\texttt{cone\\_vec}, one row per view"),
    "volume_box_mm": ("mm", "volume extent $x_0,x_1,y_0,y_1,z_0,z_1$"), "voxel_mm": ("mm", "voxel size (slice, row, column)"),
    "dose_fractions": ("--", "the three dose fractions"),
    "photons_per_pixel": ("photons", "full-dose photons per binned pixel $N_0$, from the image noise"),
    "electronic_sigma": ("photons", "electronic noise s.d. $\\sigma_e$"),
    "dm_photons_per_pixel": ("photons", "same, mammogram"), "dm_electronic_sigma": ("photons", "same, mammogram"),
    "patient": ("--", "VICTRE phantom identifier"), "density": ("--", "dense, hetero, scattered or fatty"),
    "signal_present": ("--", "true if the phantom carries lesion insertions"),
}
test_files = sorted((DATA/"test").glob("*.npz")) if (DATA/"test").exists() else []
f0 = next((f for f in test_files if ".partial" not in f.name), None)
if need("tab1_schema", f0):
    body = []
    with np.load(f0) as d:
        S = d["sart_full"].shape[0]
        for k in d.files:
            a = d[k]; sh = list(a.shape)
            if k == "lesions": sh_s = "$(n, 5)$"
            elif k == "victre_fbp": sh_s = "$(S, %d, %d)$ or $(0,0,0)$" % (VOLR, VOLC)
            elif a.ndim == 0: sh_s = "()"
            else:
                if a.ndim == 3 and sh[0] == S: sh[0] = "S"
                sh_s = "$(" + ", ".join(map(str, sh)) + ")$"
            dt = "str" if a.dtype.kind in "US" else str(a.dtype)
            unit, desc = DESC.get(k, ("--", ""))
            avail = "where published" if k == "victre_fbp" else "all"
            body.append([tt(k), sh_s, dt, unit, desc, avail])
    write_table("tab1_schema", f"{f0.parent.name}/{f0.name}",
        f"\\textbf{{Each record is one compressed \\texttt{{.npz}} file holding {len(body)} fields.}} The file key \\texttt{{patient}} is the VICTRE phantom identifier. $S$ is the number of 1\\,mm slices "
        f"({SLC['dense']}, {SLC['hetero']}, {SLC['scattered']} and {SLC['fatty']} for the dense, heterogeneous, "
        "scattered and fatty classes).",
        "tab:schema", "@{}lll>{\\raggedright\\arraybackslash}p{0.13\\textwidth}>{\\raggedright\\arraybackslash}p{0.36\\textwidth}l@{}",
        [r"Key & Shape & dtype & Unit & Description & Available \\"], body, resize=True,
        note=f"{len(body)} fields; shapes from an example file")
    NUM["Nkeys"] = str(len(body))
""")

code(r"""
# ------------------------------------------------------------ Table 2: composition
if need("tab2_composition", K):
    body = []
    def row(lab, R):
        return [lab, f"{tex_int(R.signal_present.sum())} / {tex_int((1 - R.signal_present).sum())}",
                tex_int(R.slices.sum()), tex_int(R.n_mass.sum()), tex_int(R.n_calc.sum()),
                tex_int(R.n_control.sum()), tex_int(R.victre_fbp.sum()), f"{R.size_mb.sum() / 1000:.1f}"]
    for s in ("train", "val", "test"):
        Rs = K[K.split == s]
        for i, dn in enumerate(DENS):
            body.append([({"train": "Train", "val": "Validation", "test": "Test"}[s] if i == 0 else "")] + row(DNAME[dn], Rs[Rs.density == dn]))
        body.append(["", ] + row("\\textit{all}", Rs)); body.append("MIDRULE")
    body.append(["\\textbf{Total}"] + row("", K))
    write_table("tab2_composition", "analysis/registry.csv",
        f"\\textbf{{The release holds {tex_int(len(K))} phantoms in frozen training, validation and test splits.}} Phantoms are counted as signal present / signal absent. Masses and "
        "calcification clusters are the trial's insertions in signal-present phantoms; controls are the trial's "
        "matched lesion-free positions in signal-absent phantoms. \\texttt{victre\\_fbp} counts the phantoms "
        "for which TCIA publishes the trial's own reconstruction. All counts are read from the released files.",
        "tab:composition", "llrrrrrrr",
        [r"Split & Density & Phantoms & Slices & Masses & Calc. clusters & Controls & \texttt{victre\_fbp} & Size (GB) \\"],
        body, resize=True)
""")

code(r"""
# ------------------------------------------------------------ Table 3: classical methods x dose (MC)
if need("tab3_classical", EM, ES):
    g = EM[EM.source == "mc"]; e = ES[(ES.source == "mc") & (ES.dose == "25") & (ES.kind == 1)]
    body = []
    for m in METH:
        r = [m]
        for k in DOSES:
            v = g[(g.method == m) & (g.dose == k)]
            r.append(pm(v.psnr.mean(), v.psnr.std()))
            NUM[f"PSNR{MW[m]}{DOSEW[k]}"] = f"{v.psnr.mean():.2f}"
        for k in DOSES:
            v = g[(g.method == m) & (g.dose == k)]
            r.append(f"{v.ssim.mean():.1f}"); NUM[f"SSIM{MW[m]}{DOSEW[k]}"] = f"{v.ssim.mean():.1f}"
        sm = e[e.method == m].sdnr
        r.append(f"{sm.mean():.2f}"); NUM[f"SDNRMass{MW[m]}TwentyFive"] = f"{sm.mean():.2f}"
        body.append(r)
    n = g.patient.nunique(); NUM["NtestEk"] = tex_int(n)
    tune = (f"SART uses {SETTINGS['sart_iterations']} passes over single-view subsets and SART-TV a TV step ratio "
            f"$\\alpha={SETTINGS['tv_alpha']:g}$, both chosen on validation phantoms. ") if SETTINGS else ""
    if SETTINGS:
        NUM["SARTIterations"] = str(SETTINGS["sart_iterations"]); NUM["TVAlpha"] = f"{SETTINGS['tv_alpha']:g}"
    write_table("tab3_classical", "analysis/extra_metrics.csv, extra_sdnr.csv",
        f"\\textbf{{PSNR against $x_\\mathrm{{ref}}$ falls with dose for every classical reconstructor.}} Monte Carlo projections, test split, $n={n}$ phantoms. "
        "PSNR (dB, mean $\\pm$ s.d. over phantoms) and SSIM (\\%) against $x_\\mathrm{ref}$ inside the breast "
        "mask after a least-squares gain and offset, as in \\texttt{dataset\\_test.py}; mass SDNR at 25\\,\\% dose. "
        + tune + "Table 4 gives results at 25\\,\\% dose under synthetic projections and against \\texttt{victre\\_fbp}.",
        "tab:classical", "l" + "c" * 7,
        [r" & \multicolumn{3}{c}{PSNR (dB)} & \multicolumn{3}{c}{SSIM (\%)} & mass SDNR \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-8}",
         "Method & " + " & ".join(f"{k}\\,\\%" for k in DOSES) + " & " + " & ".join(f"{k}\\,\\%" for k in DOSES) + r" & 25\,\% \\"],
        body)
""")

code(r"""
# ------------------------------------------------------------ Table 4: projection source and reference (25 %)
if need("tab4_source", EM):
    g = EM[EM.dose == "25"]
    rows = {m: dict(syn=g[(g.source == "syn") & (g.method == m)].psnr.mean(),
                    mc=g[(g.source == "mc") & (g.method == m)].psnr.mean(),
                    vf=g[(g.source == "mc") & (g.method == m)].corr_vf.dropna().mean()) for m in METH}
    if TM is not None:
        t = TM[(TM.model == "U-Net") & (TM.train_dose == "25") & (TM.dose == "25")]
        if len(t):
            rows["U-Net"] = dict(syn=t[t.source == "syn"].psnr.mean(), mc=t[t.source == "mc"].psnr.mean(),
                                 vf=t[t.source == "mc"].corr_vf.dropna().mean())
    names = list(rows)
    rk = {src: dict(zip(names, rankdata([-rows[m][src] for m in names], method="min").astype(int))) for src in ("syn", "mc")}
    body = []
    for m in names:
        r = rows[m]
        body.append([m + ("$^{a}$" if m == "U-Net" else ""), f"{r['syn']:.2f}", str(rk["syn"][m]), f"{r['mc']:.2f}",
                     str(rk["mc"][m]), (f"{r['vf']:.3f}" if np.isfinite(r["vf"]) else "--")])
        NUM[f"PSNRSyn{MW[m]}"] = f"{r['syn']:.2f}"; NUM[f"RankSyn{MW[m]}"] = str(rk["syn"][m])
        NUM[f"RankMC{MW[m]}"] = str(rk["mc"][m])
        if np.isfinite(r["vf"]): NUM[f"CorrVFBP{MW[m]}TwentyFive"] = f"{r['vf']:.3f}"
    nvf = g[(g.source == "mc") & g.corr_vf.notna()].patient.nunique(); NUM["NtestVFBPEk"] = tex_int(nvf)
    write_table("tab4_source", "analysis/extra_metrics.csv, training_metrics.csv",
        "\\textbf{Classical and supervised baselines at 25\\,\\% dose under two projection sources.} Test split. Synthetic projections are "
        "the noiseless $A x_\\mathrm{ref}$ (no gain, clipped at zero, on the breast footprint and valid rays) with noise drawn directly at the dose, mean $fN_0e^{-p}$ and variance $fN_0e^{-p}+\\sigma_e^2$; Monte Carlo projections are the trial's "
        "(\\texttt{proj\\_25}). PSNR (dB) against $x_\\mathrm{ref}$ and the rank among the methods listed. "
        f"The last column is the masked correlation with \\texttt{{victre\\_fbp}} on the $n={nvf}$ phantoms that carry it "
        "(no gain or offset needed for a correlation)."
        + (" $^{a}$U-Net trained at 25\\,\\% on Monte Carlo inputs (Supplementary Table S5)." if "U-Net" in rows else ""),
        "tab:source", "lccccc",
        [r" & \multicolumn{2}{c}{Synthetic} & \multicolumn{2}{c}{Monte Carlo} & \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"Method & PSNR & rank & PSNR & rank & correlation with \texttt{victre\_fbp} \\"], body)
""")

md("## D · Main figures")

code(r"""
# =================================================================== FIG 1 -- images for the TikZ figure
# The same file names, squares and windows as the make_thumbs.py of the Fig 1 template (fig1_pipeline):
# the crops (trim) in fig1_pipeline.tex are written for the full square images.
# Low-dose keys in v10/v11: 50/25/10. Source: the panel record of 03 (one phantom, scattered).
PK = PICK(PAN)
if need("fig1_thumbs", PK):
    p = PAN[PK]
    LCOL = {1: (0.0, 0.62, 0.45), 2: (0.84, 0.37, 0.0), 3: (0.0, 0.45, 0.70)}   # make_thumbs.LESION_COLORS
    def wnd(img, mask=None):
        v = img[mask] if mask is not None else img
        return float(np.percentile(v, 1)), float(np.percentile(v, 99.5))
    def gray(path, img, lo, hi):
        plt.imsave(path, np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1), cmap="gray", vmin=0, vmax=1)
    va = p["proj_valid"].astype(bool); full = p["proj_full"].astype(np.float32); lo, hi = wnd(full, va)
    gray(THUMB/"proj_full.png", full, lo, hi)
    gray(THUMB/"proj_view00.png", p["view00_mc"], lo, hi); gray(THUMB/"proj_view24.png", p["view24_mc"], lo, hi)
    rr, cc = np.argwhere(va).mean(0).astype(int); rows, cols = slice(rr - 80, rr + 80), slice(cc - 80, cc + 80)
    clo, chi = wnd(full[rows, cols]); gray(THUMB/"proj_full_crop.png", full[rows, cols], clo, chi)
    for k in DOSES:
        q = p[f"proj_{k}"].astype(np.float32)
        gray(THUMB/f"proj_{k}.png", q, lo, hi); gray(THUMB/f"proj_{k}_crop.png", q[rows, cols], clo, chi)
    m = p["mask"].astype(bool); g = p["sart_full"].astype(np.float32); glo, ghi = wnd(g, m)
    gray(THUMB/"sart_slice.png", g, glo, ghi); gray(THUMB/"sart_depth.png", p["depth_cut"], glo, ghi)
    gray(THUMB/"sart_mass_crop.png", p["gt_mass"], glo, ghi)
    rgb = plt.cm.gray(np.clip((g - glo) / (ghi - glo), 0, 1))[..., :3]; roi = p["lesion_roi"]
    for lab, col in LCOL.items():
        inside = roi == lab; rgb[inside] = 0.5 * rgb[inside] + 0.5 * np.array(col)
    plt.imsave(THUMB/"sart_lesion_overlay.png", rgb)
    dm = p["dm_full"].astype(np.float32); gray(THUMB/"dm_full.png", dm, *wnd(dm, p["dm_valid"].astype(bool)))
    MADE.append("fig1_thumbs"); print(f"   -> fig1_thumbs/ ({len(list(THUMB.glob('*.png')))} png, phantom {p['patient']}, {PK})")
    NUM["FigOnePatient"] = str(p["patient"])
""")

code(r"""
# =================================================================== FIG 2 -- geometry and forward model
if need("fig2_geometry", VR, PAN):
    fig = plt.figure(figsize=(W2, 4.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0], hspace=0.45, wspace=0.35)
    gsb = gs[1, :].subgridspec(1, 2, width_ratios=[0.78, 1.25], wspace=0.3)
    # (a) side view: source arc, detector, the volume at Delta
    a = fig.add_subplot(gs[0, :2]); BREAK, SQ = 130.0, 0.11
    zmap = lambda mm: np.where(np.asarray(mm, float) <= BREAK, mm, BREAK + (np.asarray(mm, float) - BREAK) * SQ)
    th = np.deg2rad(np.linspace(-ANG, ANG, NV)); sx, sy = SIDc * np.sin(th), zmap((SDDc - SIDc) + SIDc * np.cos(th))
    a.plot(sx, sy, color="#9AA5B1", lw=0.8); a.scatter(sx, sy, s=5, color=DCOL["dense"], zorder=3)
    a.text(0, sy.max() + 7, f"{NV} source positions over $\\pm${ANG:.0f}$^\\circ$", ha="center", va="bottom", fontsize=6.5)
    DW = DETR * PIX
    a.plot([-DW/2, DW/2], [0, 0], color="k", lw=2.2, solid_capstyle="butt")
    a.text(0, -12, f"detector, {PIX:.2f} mm pixels", ha="center", va="top", fontsize=6.3)
    a.add_patch(plt.Rectangle((-DW/2, -4), DW, 4, fc="#E5E5E5", ec="none", zorder=1))
    for dn in DENS:
        x0, x1, _, _, z0, z1 = volume_box(dn, SLC[dn])
        a.add_patch(plt.Rectangle((x0, z0 + (SDDc - SIDc)), x1 - x0, z1 - z0, fc="none", ec=DCOL[dn], lw=0.8, zorder=3))
    a.text(DW/2 - 4, DELTA_PUB + SLC["fatty"] + 4, "volume per density,\nlower face at $\\Delta$", ha="right", va="bottom", fontsize=6)
    for i in (0, NV // 2, NV - 1): a.plot([sx[i], 0], [sy[i], 0], color=DCOL["dense"], lw=0.4, alpha=0.35)
    for xp, m0, m1, lab in ((DW/2 + 62, 0.0, SDDc, f"SDD {SDDc:.0f} mm"), (DW/2 + 36, SDDc - SIDc, SDDc, f"SID {SIDc:.0f} mm")):
        y0, y1 = float(zmap(m0)), float(zmap(m1))
        a.annotate("", xy=(xp, y0), xytext=(xp, y1), arrowprops=dict(arrowstyle="<->", lw=0.6, color=MUTED, shrinkA=0, shrinkB=0))
        a.text(xp - 3, (y0 + y1) / 2, lab, rotation=90, ha="right", va="center", fontsize=6, color=MUTED)
    a.text(0, BREAK + 2, f"axis compressed above {BREAK:.0f} mm", ha="center", va="bottom", fontsize=6, style="italic", color=MUTED)
    a.set_xlim(-DW/2 - 20, DW/2 + 70); a.set_ylim(-25, float(zmap(SDDc)) + 25); a.axis("off"); panel(a, "a")
    # (b) top view: detector, collimated field, breast shadow, volume projections
    b = fig.add_subplot(gs[0, 2:]); p = PAN[PICK(PAN)]
    ext = (-DW/2, DW/2, -DETC * PIX / 2, DETC * PIX / 2)
    b.add_patch(plt.Rectangle((ext[0], ext[2]), DW, DETC * PIX, fc="#F2F2F2", ec="k", lw=0.8))
    uu = (np.arange(DETR) - DETR / 2 + 0.5) * PIX; vv = -DETC * PIX / 2 + (np.arange(DETC) + 0.5) * PIX
    b.contour(uu, vv, p["proj_valid"].astype(float).T, levels=[0.5], colors=[MUTED], linewidths=0.7, linestyles="--")
    b.contour(uu, vv, (p["proj_full"].astype(np.float32) > 0).astype(float).T, levels=[0.5], colors=[INK], linewidths=0.8)
    for dn in DENS:
        x0, x1, y0, y1, _, _ = volume_box(dn, SLC[dn])
        b.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none", ec=DCOL[dn], lw=0.8))
    b.text(ext[0] + 3, ext[2] - 3, "chest wall", ha="left", va="top", fontsize=6)
    b.set_aspect("equal"); b.set_xlim(ext[0] - 5, ext[1] + 5); b.set_ylim(ext[2] - 14, ext[3] + 5); b.axis("off")
    b.legend(handles=[Line2D([], [], color=INK, lw=0.8, label="breast shadow (central view)"),
                      Line2D([], [], color=MUTED, lw=0.7, ls="--", label="valid field (proj_valid)")] +
                     [Line2D([], [], color=DCOL[dn], lw=0.8, label=f"{DNAME[dn]} volume") for dn in DENS],
             loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=6, frameon=False, ncol=3)
    panel(b, "b")
    # (c) forward-model check: MC and s*A*x_ref, centre and end view, residual
    imgs = [("proj_full", "view12_syn", "proj_valid", "central view"), ("view00_mc", "view00_syn", "valid00", f"view at $-{ANG:.0f}^\\circ$")]
    sub = gsb[0].subgridspec(2, 3, wspace=0.06, hspace=0.12)
    for i, (km, ks, kv, lab) in enumerate(imgs):
        mc_, sy_ = p[km].astype(np.float32), p[ks].astype(np.float32); vv_ = p[kv].astype(bool) & (mc_ > 0)
        lo, hi = win_proj(mc_, vv_, PIXP); res = np.where(vv_, mc_ - sy_, 0); rl = float(np.percentile(np.abs(res[vv_]), 99))
        r0, r1, c0, c1 = bbox(vv_)
        for j, (img, cm, vmin, vmax, t) in enumerate(((mc_, "gray", lo, hi, "Monte Carlo $p$"), (sy_, "gray", lo, hi, "$s\\,A x_\\mathrm{ref}$"),
                                                     (res, "RdBu_r", -rl, rl, "residual"))):
            ax = fig.add_subplot(sub[i, j]); ax.imshow(img[r0:r1, c0:c1], cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest"); noaxis(ax)
            if i == 0: ax.set_title(t, fontsize=6.5)
            if j == 0: ax.set_ylabel(lab, fontsize=6.3); ax.yaxis.set_visible(True); ax.set_yticks([])
            if i == 0 and j == 0: tag(ax, "c", x=-0.35, y=1.32)
    # (d) normalised residual per view, median and quartiles per density
    d_ = fig.add_subplot(gsb[1]); ang = np.linspace(-ANG, ANG, NV)
    for dn in DENS:
        q = VR[VR.density == dn].groupby("view").res.quantile([0.25, 0.5, 0.75]).unstack()
        if not len(q): continue
        d_.fill_between(ang[q.index], q[0.25], q[0.75], color=DCOL[dn], alpha=0.18, lw=0)
        d_.plot(ang[q.index], q[0.5], color=DCOL[dn], marker=DMARK[dn], ms=2.5, lw=1.0, label=DNAME[dn])
    d_.set_xlabel("view angle ($^\\circ$)"); d_.set_ylabel("$\\|p - s\\,A x_\\mathrm{ref}\\| / \\|p\\|$")
    d_.legend(frameon=False, fontsize=6); recessive(d_); panel(d_, "d")
    save(fig, "fig2_geometry"); plt.show()
    for dn in DENS:
        v = VR[VR.density == dn].res
        if len(v): NUM[f"ViewRes{dn.capitalize()}"] = f"{v.median():.3f}"
    NUM["ViewResAll"] = f"{VR.res.median():.3f}"; NUM["ViewScale"] = f"{VR.scale.median():.3f}"
    _ang, _mins = np.linspace(-ANG, ANG, NV), []
    for dn in DENS:
        _m = VR[VR.density == dn].groupby("view").res.median()
        if len(_m): _mins += [abs(_ang[_m[_m.index < NV // 2].idxmin()]), abs(_ang[_m[_m.index > NV // 2].idxmin()])]
    if _mins: NUM["ViewResMinAngLo"], NUM["ViewResMinAngHi"] = f"{min(_mins):.1f}", f"{max(_mins):.1f}"
""")

code(r"""
# =================================================================== FIG 3 -- example records
if need("fig3_records", PAN):
    CROP3 = 20; NUM["FigThreeCropMM"] = str(CROP3)          # crop side (mm)
    rowsP = [dn for dn in DENS if dn in PAN]
    COLS = ["projection, full", "crop, 50 %", "crop, 25 %", "crop, 10 %", "$x_\\mathrm{ref}$ + lesion_roi",
            "depth cut", "victre_fbp", "mammogram"]
    fig, axs = plt.subplots(len(rowsP), len(COLS), figsize=(W2, 1.05 * len(rowsP) + 0.25),
                            gridspec_kw=dict(width_ratios=[1.1, .75, .75, .75, 1.1, 1.1, 1.1, .8]))
    axs = np.atleast_2d(axs)
    for row, dn in zip(axs, rowsP):
        p = PAN[dn]; pf, va = p["proj_full"].astype(np.float32), p["proj_valid"].astype(bool)
        lo, hi = win_proj(pf, va, PIXP); r0, r1, c0, c1 = bbox((pf > 0) & va)
        rc, cc = np.argwhere((pf > 0) & va).mean(0).astype(int); hh = int(round(CROP3 / 2 / PIXP))
        rc = int(np.clip(rc, r0 + hh, r1 - hh)); cc = int(np.clip(cc, c0 + hh + 4, c1 - hh))   # box inside the breast, not over the chest wall
        crop = (slice(rc - hh, rc + hh), slice(cc - hh, cc + hh))
        clo, chi = np.percentile(pf[crop][pf[crop] > 0], [1, 99])          # crop window: from the full dose
        ax = row[0]; ax.imshow(pf[r0:r1, c0:c1], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        ax.add_patch(plt.Rectangle((cc - hh - c0, rc - hh - r0), 2 * hh, 2 * hh, fill=False, ec="#F0E442", lw=0.7))
        scalebar(ax, PIXP, r1 - r0, c1 - c0)
        for j, k in enumerate(DOSES, 1):
            row[j].imshow(p[f"proj_{k}"].astype(np.float32)[crop], cmap="gray", vmin=clo, vmax=chi, interpolation="nearest")
        m = p["mask"].astype(bool); g = p["sart_full"].astype(np.float32); glo, ghi = win_recon(g, m); R0, R1, C0, C1 = bbox(m)
        ax = row[4]; ax.imshow(np.where(m, g, glo)[R0:R1, C0:C1], cmap="gray", vmin=glo, vmax=ghi, interpolation="nearest")
        for lab, col in LESCOL.items():
            roi = (p["lesion_roi"] == lab)[R0:R1, C0:C1]
            if roi.any(): ax.contour(roi.astype(float), levels=[0.5], colors=[col], linewidths=0.7)
        scalebar(ax, PIX, R1 - R0, C1 - C0)
        dc = np.where(p["depth_mask"].astype(bool), p["depth_cut"].astype(np.float32), glo)[:, C0:C1]
        row[5].imshow(dc, cmap="gray", vmin=glo, vmax=ghi, interpolation="nearest", aspect=1.0 / PIX)
        if bool(p["victre_fbp_present"]):
            vf = p["victre_fbp"].astype(np.float32); vlo, vhi = win_recon(vf, m)
            row[6].imshow(np.where(m, vf, vlo)[R0:R1, C0:C1], cmap="gray", vmin=vlo, vmax=vhi, interpolation="nearest")
        else:
            row[6].text(0.5, 0.5, "not released\nfor this phantom", ha="center", va="center", fontsize=6, color=MUTED, transform=row[6].transAxes)
        dm, dv = p["dm_full"].astype(np.float32), p["dm_valid"].astype(bool); mr0, mr1, mc0, mc1 = bbox((dm > 0) & dv, pad=8)
        row[7].imshow(dm[mr0:mr1, mc0:mc1], cmap="gray", vmin=win_proj(dm, dv, PIXM)[0], vmax=win_proj(dm, dv, PIXM)[1], interpolation="nearest")
        for ax in row: noaxis(ax)
        row[0].set_ylabel(f"{DNAME[dn]}\n{int(p['slices'])} slices", fontsize=6.5); row[0].yaxis.set_visible(True)
    fig.legend(handles=[Line2D([], [], color=LESCOL[2], lw=1, label="mass"), Line2D([], [], color=LESCOL[1], lw=1, label="calcification cluster"),
                        Line2D([], [], color="#F0E442", lw=1, label="crop region")],
               loc="lower center", ncol=3, frameon=False, fontsize=6, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(w_pad=0.15, h_pad=0.25, rect=(0, 0.03, 1, 0.97))
    ytop = max(ax.get_position().y1 for ax in axs[0])            # titles on one line: above the highest panel
    for ax, t in zip(axs[0], COLS):
        bb = ax.get_position(); fig.text((bb.x0 + bb.x1) / 2, ytop + 0.006, t, ha="center", va="bottom", fontsize=6.2)
    save(fig, "fig3_records"); plt.show()
    for dn in rowsP: NUM[f"FigThree{dn.capitalize()}"] = str(PAN[dn]["patient"])
""")

code(r"""
# =================================================================== FIG 4 -- dataset statistics
# Lesion depth from DIDSR's own .loc files (small download); generation verified that the crop drops
# no position, so the .loc positions are the lesions in the records.
LOC_URL = "https://raw.githubusercontent.com/DIDSR/VICTRE/master/Locations/DBT-{}.tar.gz"
LOC = []
if K is not None:
    ids = set(K.patient.astype(str).str.lstrip("-"))
    for dn in DENS:
        try:
            raw = urllib.request.urlopen(LOC_URL.format(dn), timeout=60).read()
        except Exception as e:
            print("   .loc could not be downloaded:", dn, e); LOC = []; break
        with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
            for mem in tf.getmembers():
                nm = Path(mem.name).name
                if not (nm.startswith("roi_S") and nm.endswith(".loc")): continue
                kind, seed = nm[4:6], nm[7:-4].lstrip("-")
                if seed not in ids: continue
                for ln in tf.extractfile(mem).read().decode().splitlines():
                    v = ln.split()
                    if len(v) == 4:
                        num = int(float(v[3])); typ = ("calc." if num < 4 else "mass") if kind == "SP" else "control"
                        LOC.append(dict(density=dn, type=typ, zfrac=float(v[2]) / SLC[dn]))
LOC = pd.DataFrame(LOC)
if need("fig4_statistics", K):
    fig, axs = plt.subplots(2, 3, figsize=(W2, 3.9))
    # (a) phantoms: density x signal x split
    a = axs[0, 0]; x = np.arange(len(DENS)); bottom = np.zeros(len(DENS))
    HATCH = {"train": "", "val": "////", "test": "...."}
    for s in ("train", "val", "test"):
        for sp, al in ((1, 1.0), (0, 0.45)):
            h = np.array([((K.density == dn) & (K.split == s) & (K.signal_present == sp)).sum() for dn in DENS])
            a.bar(x, h, bottom=bottom, color=[DCOL[dn] for dn in DENS], alpha=al, hatch=HATCH[s], ec="white", lw=0.3)
            bottom += h
    a.set_xticks(x); a.set_xticklabels([DSHORT[d] for d in DENS]); a.set_ylabel("phantoms"); recessive(a)
    a.legend(handles=[mpl.patches.Patch(fc="#888888", label="signal present"), mpl.patches.Patch(fc="#888888", alpha=0.45, label="signal absent")] +
                     [mpl.patches.Patch(fc="white", ec=INK, hatch=HATCH[s], label=s) for s in ("train", "val", "test")],
             fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=3); panel(a, "a")
    # (b) slice count (= compressed thickness, 1 mm slices)
    a = axs[0, 1]
    a.bar(x, [SLC[dn] for dn in DENS], color=[DCOL[dn] for dn in DENS])
    for i, dn in enumerate(DENS): a.text(i, SLC[dn] + 1, str(SLC[dn]), ha="center", va="bottom", fontsize=6)
    a.set_xticks(x); a.set_xticklabels([DSHORT[d] for d in DENS]); a.set_ylabel("slices (1 mm) = volume depth (mm)"); recessive(a); panel(a, "b")
    # (c) N0, by density (thickness)
    a = axs[0, 2]; rng = np.random.default_rng(0)
    for i, dn in enumerate(DENS):
        v = K[K.density == dn].photons.values
        a.scatter(SLC[dn] + rng.uniform(-1.2, 1.2, len(v)), v, s=2, color=DCOL[dn], marker=DMARK[dn], alpha=0.35, lw=0)
        a.plot([SLC[dn] - 1.8, SLC[dn] + 1.8], [np.median(v)] * 2, color=INK, lw=1.3)
    a.set_xlabel("volume depth (mm)"); a.set_ylabel("photons per binned pixel, $N_0$"); recessive(a); panel(a, "c")
    # (d) lesions per phantom, by type
    a = axs[1, 0]; TY = [("n_mass", "mass", 1), ("n_calc", "calc. cluster", 1), ("n_control", "control", 0)]
    for j, (col, lab, sp) in enumerate(TY):
        for i, dn in enumerate(DENS):
            v = K[(K.density == dn) & (K.signal_present == sp)][col].values
            if not len(v): continue
            cnt = pd.Series(v).value_counts().sort_index()
            xs = j * 5 + i
            a.scatter([xs] * len(cnt), cnt.index, s=np.clip(cnt.values, 4, 60), color=DCOL[dn], marker=DMARK[dn], alpha=0.8, lw=0)
    a.set_xticks([1.5, 6.5, 11.5]); a.set_xticklabels([t[1] for t in TY]); a.set_ylabel("lesions per phantom"); recessive(a); panel(a, "d")
    # (e) lesion depth / thickness, by type
    a = axs[1, 1]
    if len(LOC):
        for j, typ in enumerate(("mass", "calc.", "control")):
            for i, dn in enumerate(DENS):
                v = LOC[(LOC.density == dn) & (LOC.type == typ)].zfrac.values
                if not len(v): continue
                bp = a.boxplot([v], positions=[j * 5 + i], widths=0.7, patch_artist=True, showfliers=False)
                for b_ in bp["boxes"]: b_.set(facecolor=DCOL[dn], alpha=0.6, lw=0.5)
                for k_ in ("whiskers", "caps", "medians"):
                    for l_ in bp[k_]: l_.set(color=INK, lw=0.6)
        a.set_xticks([1.5, 6.5, 11.5]); a.set_xticklabels(["mass", "calc. cluster", "control"])
    else:
        a.text(0.5, 0.5, ".loc files not reachable", ha="center", va="center", transform=a.transAxes, color=MUTED)
    a.set_ylabel("depth / volume depth"); a.set_ylim(0, 1); recessive(a); panel(a, "e")
    # (f) availability of victre_fbp
    a = axs[1, 2]
    pc = [100 * K[K.density == dn].victre_fbp.mean() for dn in DENS]
    a.bar(x, pc, color=[DCOL[dn] for dn in DENS])
    for i, v in enumerate(pc): a.text(i, v + 1.5, f"{v:.0f} %", ha="center", va="bottom", fontsize=6)
    a.set_xticks(x); a.set_xticklabels([DSHORT[d] for d in DENS]); a.set_ylim(0, 110); a.set_ylabel("phantoms with victre_fbp (%)")
    recessive(a); panel(a, "f")
    fig.legend(handles=[Line2D([], [], ls="", marker=DMARK[dn], color=DCOL[dn], label=DNAME[dn]) for dn in DENS],
               loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(w_pad=1.0, h_pad=2.2, rect=(0, 0, 1, 0.96)); save(fig, "fig4_statistics"); plt.show()
    NUM["PhotonsMedian"] = tex_int(round(K.photons.median())); NUM["PhotonsMin"] = tex_int(round(K.photons.min()))
    NUM["PhotonsMax"] = tex_int(round(K.photons.max()))
    for dn in DENS: NUM[f"PctVFBP{dn.capitalize()}"] = f"{100 * K[K.density == dn].victre_fbp.mean():.0f}"
    if len(LOC): NUM["NLocPositions"] = tex_int(len(LOC))
""")

code(r"""
# =================================================================== FIG 5 -- dose fidelity
if need("fig5_dose", NB, NP, B, PAN):
    fig = plt.figure(figsize=(W2, 3.9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0], hspace=0.62, wspace=0.75)
    # (a) measured extra variance vs model, bins by count (median over phantoms)
    a = fig.add_subplot(gs[0, 0])
    for k in DOSES:
        q = NB[NB.dose == k].groupby("bin")[["measured", "predicted"]].median()
        a.loglog(q.predicted, q.measured, marker=DOSEMARK[k], color=DOSECOL[k], mec=INK, mew=0.3, ms=3.5, lw=0.9, label=DLAB[k])
    lim = [NB[["measured", "predicted"]].clip(lower=1e-12).min().min(), NB[["measured", "predicted"]].max().max()]
    a.plot(lim, lim, color=INK, lw=0.6, ls="--", zorder=5, alpha=0.7)            # above the data: the points sit on the line
    a.set_xlabel("predicted extra variance"); a.set_ylabel("measured extra variance"); a.legend(frameon=False, title="dose", title_fontsize=6, fontsize=6, handlelength=1.3, loc="lower right", borderaxespad=0.1)
    panel(a, "a")
    # (b) NPS, OS-SART, per dose (mean over phantoms)
    a = fig.add_subplot(gs[0, 1])
    for k in DOSES:
        q = NP[(NP.method == "OS-SART") & (NP.dose == k)].groupby("freq").nps.mean()
        q = q[q.index > 0]
        a.semilogy(q.index, q.values, color=DOSECOL[k], marker=DOSEMARK[k], mec=INK, mew=0.3, ms=2.5, lw=1.0, label=DLAB[k])
    a.set_xlabel("spatial frequency (mm$^{-1}$)"); a.set_ylabel("NPS, OS-SART"); recessive(a); panel(a, "b")
    # (c) PSNR density x dose (01, OS-SART = the dataset's recipe, whole test split)
    a = fig.add_subplot(gs[0, 2:]); g = B[(B.method == "SART") & (B.ref == "sart_full")]
    for j, k in enumerate(DOSES):
        for i, dn in enumerate(DENS):
            v = g[(g.density == dn) & (g.dose == k)].psnr.values
            if not len(v): continue
            bp = a.boxplot([v], positions=[j * 5 + i], widths=0.75, patch_artist=True, showfliers=False)
            for b_ in bp["boxes"]: b_.set(facecolor=DCOL[dn], alpha=0.9, lw=0.5)
            for k_ in ("whiskers", "caps", "medians"):
                for l_ in bp[k_]: l_.set(color=INK, lw=0.6)
    a.set_xticks([1.5, 6.5, 11.5]); a.set_xticklabels([DLAB[k] for k in DOSES]); a.set_xlabel("dose")
    a.set_ylabel("PSNR of OS-SART vs $x_\\mathrm{ref}$ (dB)"); recessive(a)
    a.legend(handles=[mpl.patches.Patch(fc=DCOL[dn], alpha=0.65, label=DNAME[dn]) for dn in DENS], frameon=False, fontsize=6, ncol=2, loc="lower left")
    panel(a, "c")
    # (d) mammogram crop, per dose (same window)
    p = PAN[PICK(PAN)]; dm, dv = p["dm_full"].astype(np.float32), p["dm_valid"].astype(bool)
    rc, cc = np.argwhere((dm > 0) & dv).mean(0).astype(int); hh = int(round(15 / PIXM))
    cr = (slice(rc - hh, rc + hh), slice(cc - hh, cc + hh)); lo, hi = np.percentile(dm[cr][dm[cr] > 0], [1, 99])
    sub = gs[1, :].subgridspec(1, 4, wspace=0.04)
    for j, k in enumerate(DOSE_ALL):
        ax = fig.add_subplot(sub[0, j]); img = (dm if k == "100" else p[f"dm_{k}"].astype(np.float32))[cr]
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest"); noaxis(ax); ax.set_title(f"mammogram, {DLAB[k]}", fontsize=6.5)
        scalebar(ax, PIXM, 2 * hh, 2 * hh)
        if j == 0: tag(ax, "d", x=-0.08, y=1.12)
    save(fig, "fig5_dose"); plt.show()
    for k in DOSES:
        q = NB[NB.dose == k]; NUM[f"NoiseBinRatio{DOSEW[k]}"] = f"{(q.measured / q.predicted).median():.3f}"
""")

code(r"""
# =================================================================== FIG 6 -- methods
if need("fig6_methods", EM, PAN):
    PKm = next((dn for dn in ("hetero", "scattered", "dense", "fatty") if dn in PAN), None); p = PAN[PKm]
    m = p["mask"].astype(bool); g = p["sart_full"].astype(np.float32); lo, hi = win_recon(g, m); R0, R1, C0, C1 = bbox(m)
    # Image and error as in the metrics: least-squares gain + offset inside the mask (dataset_test.py), then one window.
    def fitg(img):
        x = img[m].astype(np.float64); A_ = np.c_[x, np.ones_like(x)]
        (a_, b_), *_ = np.linalg.lstsq(A_, g[m].astype(np.float64), rcond=None)
        return (a_ * img + b_).astype(np.float32)
    FIT = {(mm, k): fitg(p[f"slice_{MKEY[mm]}_{k}"].astype(np.float32)) for mm in METH for k in DOSES}
    err_l = float(np.percentile(np.abs(np.concatenate([(v - g)[m] for v in FIT.values()])), 99))
    fig = plt.figure(figsize=(W2, 5.7))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.0], hspace=0.2, bottom=0.1)
    top = gs[0].subgridspec(len(DOSES), 2 * len(METH) + 3, wspace=0.03, hspace=0.05,
                            width_ratios=[1] * len(METH) + [0.12] + [1] * len(METH) + [0.06, 0.08])
    for i, k in enumerate(DOSES):
        for j, mm in enumerate(METH):
            img = FIT[(mm, k)]
            ax = fig.add_subplot(top[i, j]); ax.imshow(np.where(m, img, lo)[R0:R1, C0:C1], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
            noaxis(ax)
            if i == 0: ax.set_title(mm, fontsize=6.5)
            if j == 0: ax.set_ylabel(DLAB[k], fontsize=6.5); ax.yaxis.set_visible(True)
            if i == 0 and j == 0: tag(ax, "a", x=-0.25, y=1.2)
            if i == len(DOSES) - 1 and j == 0: scalebar(ax, PIX, R1 - R0, C1 - C0)
            ax = fig.add_subplot(top[i, len(METH) + 1 + j])
            imE = ax.imshow(np.where(m, img - g, 0)[R0:R1, C0:C1], cmap="RdBu_r", vmin=-err_l, vmax=err_l, interpolation="nearest"); noaxis(ax)
            if i == 0: ax.set_title(mm, fontsize=6.5)
            if i == 0 and j == 0: tag(ax, "b", x=-0.12, y=1.2)
    cax = fig.add_subplot(top[:, -1])
    cb = fig.colorbar(imE, cax=cax); cb.ax.tick_params(labelsize=6, length=2)
    cb.set_label("error vs $x_\\mathrm{ref}$ (mm$^{-1}$)", fontsize=6)
    bot = gs[1].subgridspec(1, 3, wspace=0.42)
    XD = {"10": 0, "25": 1, "50": 2}
    for c_, src, L in ((0, "mc", "c"), (1, "syn", "")):
        a = fig.add_subplot(bot[0, c_])
        for mm in METH:
            v = EM[(EM.source == src) & (EM.method == mm)].groupby("dose").psnr.agg(["mean", "std"]).reindex(["10", "25", "50"])
            a.errorbar([XD[k] for k in v.index], v["mean"], yerr=v["std"], color=INK, ms=3.5, capsize=1.5, lw=1.0, label=mm, **MSTY[mm])
        a.set_xticks(range(3)); a.set_xticklabels(["10", "25", "50"]); a.set_xlabel("dose (%)")
        a.set_ylabel("PSNR vs $x_\\mathrm{ref}$ (dB)"); a.set_title(SRCLAB[src] + " projections", fontsize=6.5); recessive(a)
        if L: panel(a, L)
    # (d) rank, two sources (25 %)
    a = fig.add_subplot(bot[0, 2]); q = EM[EM.dose == "25"].groupby(["source", "method"]).psnr.mean().unstack(0)
    rs = {src: dict(zip(q.index, rankdata(-q[src].values, method="min"))) for src in ("syn", "mc")}
    for mm in METH:
        a.plot([0, 1], [rs["syn"][mm], rs["mc"][mm]], color=INK, ms=4, lw=1.0, **MSTY[mm])
        a.text(-0.08, rs["syn"][mm], mm, ha="right", va="center", fontsize=6); a.text(1.08, rs["mc"][mm], mm, ha="left", va="center", fontsize=6)
    a.set_xlim(-0.7, 1.7); a.set_ylim(len(METH) + 0.5, 0.5); a.set_xticks([0, 1]); a.set_xticklabels(["synthetic", "Monte Carlo"])
    a.set_yticks([]); a.set_ylabel("rank by PSNR at 25 % (top = 1)"); a.spines["left"].set_visible(False); panel(a, "d")
    fig.legend(handles=[Line2D([], [], color=INK, lw=1.0, ms=3.5, label=mm, **MSTY[mm]) for mm in METH],
               loc="lower center", ncol=len(METH), frameon=False, fontsize=6, bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig6_methods"); plt.show()
    NUM["FigSixDensity"] = DNAME[PKm]; NUM["FigSixPatient"] = str(p["patient"])
""")

code(r"""
# =================================================================== FIG 7 -- lesions
if need("fig7_lesions", ES, PAN):
    p = PAN[next((dn for dn in ("hetero", "scattered", "dense", "fatty") if dn in PAN))]
    gm = p["gt_mass"].astype(np.float32); mlo, mhi = np.percentile(gm[gm > 0], [2, 98])
    gc = p["gt_calc"].astype(np.float32); clo, chi = np.percentile(gc[gc > 0], [2, 99.5])
    fig = plt.figure(figsize=(W2, 5.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.55, 1.0], hspace=0.2)
    top = gs[0].subgridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.12)
    def crop_grid(spec, ref, keyfmt, doses, lo_, hi_, L, mm_bar):
        # column 0: x_ref, column 1: spacer (method names here), then the doses
        G_ = spec.subgridspec(len(METH), len(doses) + 2, wspace=0.03, hspace=0.05, width_ratios=[1, 0.22] + [1] * len(doses))
        for i, mm in enumerate(METH):
            if i == 0:
                ax = fig.add_subplot(G_[i, 0]); ax.imshow(ref, cmap="gray", vmin=lo_, vmax=hi_); ax.set_title("$x_\\mathrm{ref}$", fontsize=6.5)
                scalebar(ax, PIX, *ref.shape, mm=mm_bar); tag(ax, L, x=-0.12, y=1.18); noaxis(ax)
            for j, k in enumerate(doses, 2):
                ax = fig.add_subplot(G_[i, j]); ax.imshow(p[keyfmt.format(MKEY[mm], k)], cmap="gray", vmin=lo_, vmax=hi_); noaxis(ax)
                if i == 0: ax.set_title(DLAB[k], fontsize=6.5)
                if j == 2: ax.set_ylabel(mm, fontsize=6.3); ax.yaxis.set_visible(True)
    crop_grid(top[0], gm, "mass_{}_{}", DOSES, mlo, mhi, "a", 10)
    crop_grid(top[1], gc, "calc_{}_{}", ("50", "10"), clo, chi, "b", 5)
    bot = gs[1].subgridspec(1, 3, wspace=0.42)
    # (c) depth profile from the mass centre, 25 % MC; each curve normalised to its own far background
    a = fig.add_subplot(bot[0, 0]); zc = int(round(float(p["mass"][0]))); zz = np.arange(len(p["prof_gt"])) - zc
    def norm(v):
        v = np.asarray(v, float); base = np.median(v[np.abs(zz) >= 8]) if (np.abs(zz) >= 8).any() else v.min()
        return (v - base) / max(v[zc] - base, 1e-12)
    a.plot(zz, norm(p["prof_gt"]), color=INK, lw=1.6, label="$x_\\mathrm{ref}$")
    # victre_fbp is not drawn here: the panel shows the depth spread of the methods (scope of the paper).
    for mm in METH: a.plot(zz, norm(p[f"prof_{MKEY[mm]}_25"]), color=INK, lw=0.8, ms=2.5, label=f"{mm}, 25 %", **MSTY[mm])
    a.axvline(0, color=GRID, lw=0.6); a.set_xlim(-12, 12); a.set_xlabel("slice relative to stored $z$ (mm)")
    a.set_ylabel("normalized core contrast"); a.legend(frameon=False, fontsize=6); recessive(a); panel(a, "c")
    # (d) depth offset of x_ref relative to victre_fbp (tissue cross-correlation, 06), per density.
    # Before: the record before the DELTA patch (log of 07, every patched file; for fatty, the lesion measurement
    # of the pilot run before the fatty correction). After: the released record (06, test split). The per-lesion peak
    # estimator is not used: it sticks to the edge of the +-3 slice window (05 B).
    a = fig.add_subplot(bot[0, 1]); a.axhline(0, color=MUTED, lw=0.7, ls="--")
    for i, dn in enumerate(DENS):
        if PATCH is not None and "k_old" in PATCH:
            v = PATCH[(PATCH.density == dn)].k_old.dropna()
            if len(v): a.errorbar(i - 0.14, v.mean(), yerr=v.std(), fmt=DMARK[dn], ms=4, color=DCOL[dn], mfc="white", capsize=2, lw=0.8)
        if dn == "fatty":
            g_ = PILOT_FATTY["gt"]
            a.errorbar(i - 0.14, g_[0], yerr=1.96 * g_[1], fmt=DMARK[dn], ms=4, color=DCOL[dn], mfc="white", capsize=2, lw=0.8, ls="none")
            a.text(i - 0.14, g_[0] + 1.96 * g_[1] + 0.04, "lesion-based", fontsize=6, color=MUTED, ha="center", va="bottom")
        if ZS is not None:
            v = ZS[(ZS.density == dn) & ~ZS.edge.astype(bool)].k
        elif PATCH is not None and "k_new" in PATCH:
            v = PATCH[PATCH.density == dn].k_new.dropna()
        else: v = pd.Series(dtype=float)
        if len(v): a.errorbar(i + 0.14, v.mean(), yerr=v.std(), fmt=DMARK[dn], ms=4, color=DCOL[dn], capsize=2, lw=0.8)
    a.legend(handles=[Line2D([], [], ls="", marker="o", mfc="white", mec=INK, color=INK, label="earlier per-class $\\Delta$"),
                      Line2D([], [], ls="", marker="o", color=INK, label="released")], frameon=False, fontsize=6,
             loc="lower left", markerscale=0.8)
    a.set_xticks(range(4)); a.set_xticklabels([DSHORT[d] for d in DENS]); a.set_xlim(-0.6, 3.5)
    a.set_ylabel("depth offset of $x_\\mathrm{ref}$ vs victre_fbp (slices)")
    recessive(a); panel(a, "d")
    # (e) mass SDNR vs dose, per method; control locations are the null hypothesis
    a = fig.add_subplot(bot[0, 2]); XD = {"10": 0, "25": 1, "50": 2}
    for mm in METH:
        v = ES[(ES.source == "mc") & (ES.method == mm) & (ES.kind == 1)].groupby("dose").sdnr.agg(["mean", "sem"]).reindex(["10", "25", "50"])
        a.errorbar([XD[k] for k in v.index], v["mean"], yerr=1.96 * v["sem"], color=INK, ms=3.5, capsize=1.5, lw=1.0, label=mm, **MSTY[mm])
        c = ES[(ES.source == "mc") & (ES.method == mm) & (ES.kind == -1)].groupby("dose").sdnr.mean().reindex(["10", "25", "50"])
        a.plot([XD[k] for k in c.index], c.values, color=MUTED, lw=0.6, **{**MSTY[mm], "marker": None})
    gt_m = ES[(ES.method == "GT") & (ES.kind == 1)].sdnr.mean()
    a.axhline(gt_m, color=INK, lw=0.6, ls=(0, (4, 2))); a.text(0.0, gt_m, "$x_\\mathrm{ref}$", fontsize=6, va="bottom", ha="left")
    a.axhline(0, color=GRID, lw=0.6)
    a.set_xticks(range(3)); a.set_xticklabels(["10", "25", "50"]); a.set_xlabel("dose (%)"); a.set_ylabel("mass SDNR (grey: controls)")
    a.legend(frameon=False, fontsize=6, loc="lower right", bbox_to_anchor=(1.0, 0.06)); recessive(a); panel(a, "e")
    save(fig, "fig7_lesions"); plt.show()
""")

md(r"""
## E · Supplementary material and extra numbers
""")
code(r"""
# noise: re-injection with the same dose model gives the same ratio (05 E)
if NRG is not None:
    NUM["NRegen"] = str(NRG.patient.nunique())
    q = NRG.pivot_table(index=["patient", "key"], columns="source", values="ratio")
    if {"stored", "regen_f16"} <= set(q.columns):
        d_ = q["stored"] - q["regen_f16"]
        NUM["NoiseRegenDiff"] = f"{d_.mean():+.3f}"; NUM["NoiseRegenDiffSD"] = f"{d_.std():.3f}"
if SD is not None:
    v = SD[(SD.method == "GT") & (SD.kind == 0)].sdnr; w = SD[(SD.method == "GT") & (SD.kind == 1)].sdnr
    if len(v): NUM["SDNRCalcRef"] = f"{v.mean():.3f}"
    if len(w): NUM["SDNRMassRef"] = f"{w.mean():.2f}"
# ------------------------------------------------------------ S1: Table 3 by density
if need("tabS1_classical_density", EM):
    g = EM[EM.source == "mc"]; body = []
    for dn in DENS:
        for i, mm in enumerate(METH):
            r = [DNAME[dn] if i == 0 else "", mm]
            for k in DOSES:
                v = g[(g.density == dn) & (g.method == mm) & (g.dose == k)].psnr
                r.append(pm(v.mean(), v.std()) if len(v) else "--")
            body.append(r)
        body.append("MIDRULE")
    body = body[:-1]
    write_table("tabS1_classical_density", "analysis/extra_metrics.csv",
        "\\textbf{PSNR against $x_\\mathrm{ref}$ falls with dose for every classical reconstructor in every density class.} Table 3 by density class. PSNR (dB, mean $\\pm$ s.d. over phantoms) against $x_\\mathrm{ref}$, Monte Carlo projections, test split.",
        "tab:S1", "llccc", ["Density & Method & " + " & ".join(f"{k}\\,\\%" for k in DOSES) + r" \\"], body)

# ------------------------------------------------------------ S2: geometry constants
rows = [("Views", f"{NV}, uniform over $\\pm{ANG:g}^\\circ$"), ("Source to rotation axis / detector", f"{SIDc:g} / {SDDc:g} mm"),
        ("Detector", f"{DETR} $\\times$ {DETC} binned pixels of {PIX:.2f} mm"), ("Volume grid", f"$S \\times {VOLR} \\times {VOLC}$, {PIX:.2f} $\\times$ {PIX:.2f} $\\times$ 1 mm"),
        ("OS-SART for $x_\\mathrm{ref}$", f"{CFGJ.get('sart_iterations', 10)} iterations $\\times$ {CFGJ.get('sart_subsets', 5)} subsets, non-negative, "
                                         f"{CFGJ.get('sart_depth_margin_slices', 8)} margin slices above and below")]
rows += [(f"{DNAME[dn]}: slices, $\\Delta$", f"{SLC[dn]}, {DELTA[dn]:.3f} mm") for dn in DENS]
rows += [("$\\Delta$ from the trial's code", f"20.0 mm object-to-detector distance less 0.5 mm, half the cover: {DELTA_PUB:.3f} mm")]
write_table("tabS2_geometry", "dataset_generate.py, config.json", "\\textbf{All density classes share one acquisition geometry and the same detector offset $\\Delta$.} Acquisition geometry and reconstruction constants of the release.",
            "tab:S2", "@{}ll@{}", [r"Quantity & Value \\"], [[a, b] for a, b in rows])

# ------------------------------------------------------------ S3: training table
if need("tabS3_training", TM, TS):
    body = []
    for name, td in [("OS-SART (input)", "")] + [("U-Net", t) for t in sorted(TM[TM.model == "U-Net"].train_dose.unique(), key=lambda s: -int(s))]:
        g = TM[(TM.model == ("OS-SART" if td == "" else "U-Net")) & (TM.train_dose == td) & (TM.source == "mc")]
        s = TS[(TS.model == ("OS-SART" if td == "" else "U-Net")) & (TS.train_dose == td) & (TS.source == "mc") & (TS.kind == 1)]
        r = [name if td == "" else f"U-Net, trained at {td}\\,\\%"]
        for k in DOSES:
            v = g[g.dose == k]; r.append(f"{v.psnr.mean():.2f} / {v.ssim.mean():.1f}")
        r.append(f"{s[s.dose == '25'].sdnr.mean():.2f}"); body.append(r)
        if td:
            for k in DOSES: NUM[f"PSNRUNet{DOSEW[td]}On{DOSEW[k]}"] = f"{g[g.dose == k].psnr.mean():.2f}"
    write_table("tabS3_training", "analysis/training_metrics.csv, training_sdnr.csv",
        "\\textbf{A supervised U-Net trained on the training split, scored on the test split.} A 2D U-Net maps the low-dose OS-SART reconstruction to $x_\\mathrm{ref}$ "
        "(masked squared error, residual learning), trained on the training split. "
        "PSNR (dB) / SSIM (\\%) against $x_\\mathrm{ref}$ at each test dose, Monte Carlo projections; mass SDNR at 25\\,\\%. "
        "The input row scores the network's input on the same phantoms.",
        "tab:S3", "lcccc", ["Model & " + " & ".join(f"{k}\\,\\%" for k in DOSES) + r" & mass SDNR \\"], body)

# ------------------------------------------------------------ S4: detection AUC (01, all volumes)
TASK = [("GT", "100", r"$x_\mathrm{ref}$"), ("VFBP", "100", r"\texttt{victre\_fbp}")] + \
       [(mm, d, f"{'OS-SART' if mm == 'SART' else mm}, {DLAB[d]}") for mm in ("SART", "FBP") for d in DOSE_ALL if not (mm == "SART" and d == "100")]
if need("tabS4_task", SD):
    body = []
    for mm, d, lab in TASK:
        g = SD[(SD.method == mm) & (SD.dose == d)]
        if not len(g): continue
        a, lo, hi, nm, nc = auc_ci(g)
        body.append([lab.replace(" %", "\\,\\%"), f"{g[g.kind == 1].sdnr.mean():.2f}",
                     f"${g[g.kind == -1].sdnr.mean():+.3f}$", f"{a:.3f} [{lo:.3f}, {hi:.3f}]", f"{nm}/{nc}"])
        NUM[f"AUC{mm}{DOSEW[d]}"] = f"{a:.3f}"; NUM[f"SDNRCtrl{mm}{DOSEW[d]}"] = f"{g[g.kind == -1].sdnr.mean():+.3f}"
        if mm == "VFBP": body.append("MIDRULE")
    write_table("tabS4_task", "analysis/sdnr.csv",
        "\\textbf{At the stored positions, mass SDNR exceeds that of matched controls in every volume; controls give SDNR near zero.} Test split. SDNR: mean of a $3\\times9\\times9$-voxel core minus the mean of a $25\\times25$ window in the centre slice (core included), "
        "over the s.d. of that window, at each stored position. AUC: masses against controls, "
        "95\\,\\% interval from a bootstrap over phantoms. The \\texttt{victre\\_fbp} row uses the phantoms that carry it. "
        f"Calcification clusters are not visible on the {PIX:.2f}\\,mm grid (Fig. 7b, Supplementary Fig. S2); their positions are stored "
        "but no detectability is reported for them.",
        "tab:S4", "lcccc", [r"Volume & mass SDNR & control SDNR & AUC [95\,\% CI] & $n$ mass/control \\"], body)

# ------------------------------------------------------------ S5: noise, all files
rows_n, src = [], None
if VAL:
    for mt in re.finditer(r"^\|\s*((?:proj|dm)_\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", VAL, re.M):
        rows_n.append((mt.group(1), float(mt.group(2)), float(mt.group(3)), float(mt.group(4))))
    src = "VICTRE-PAIRED-v11/validation.md (all files)"
if not rows_n and NZ is not None:
    rows_n = [(k, g.ratio.mean(), g.ratio.min(), g.ratio.max()) for k, g in NZ.groupby("key")]; src = "noise.csv (test)"
if need("tabS5_noise", rows_n):
    order = [f"proj_{d}" for d in DOSES] + [f"dm_{d}" for d in DOSES]
    rows_n = sorted(rows_n, key=lambda r: order.index(r[0]) if r[0] in order else 99)
    write_table("tabS5_noise", src,
        "\\textbf{The injected noise variance is close to the model prediction in every low-dose array.} Measured extra variance in each low-dose array over the variance the "
        "noise model predicts from the stored photon count and electronic noise (target 1), as the ratio of the median measured to the median predicted variance over the pixels of each file. "
        "The check verifies the injection for the stored $N_0$ and $\\sigma_e$; it does not test $N_0$ itself. "
        "Re-injecting the model into the same arrays reproduces the shortfall below 1 (Technical Validation).",
        "tab:S5", "lccc", [r"Array & mean & min & max \\"], [[tt(k), f"{a:.3f}", f"{lo:.3f}", f"{hi:.3f}"] for k, a, lo, hi in rows_n])
    NUM["NoiseMin"] = f"{min(r[2] for r in rows_n):.2f}"; NUM["NoiseMax"] = f"{max(r[3] for r in rows_n):.2f}"
    NUM["NoiseMeanMin"] = f"{min(r[1] for r in rows_n):.3f}"; NUM["NoiseMeanMax"] = f"{max(r[1] for r in rows_n):.3f}"
""")

code(r"""
# =================================================================== Fig S1 -- training
if need("figS1_training", TM, TC):
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.2), gridspec_kw=dict(width_ratios=[1.1, 1.0, 1.6], wspace=0.35))
    a = axs[0]
    for td, g in TC.groupby("dose"):
        td = str(td); a.plot(g.step, g.val_psnr, color=DOSECOL[td], marker=DOSEMARK[td], mec=INK, mew=0.3, ms=2.5, label=f"U-Net, trained at {DLAB[td]}")
        a.axhline(g.val_psnr_input.iloc[-1], color=DOSECOL[td], lw=0.7, ls="--", label=f"OS-SART input, {DLAB[td]}")
    a.set_xlabel("training step"); a.set_ylabel("validation PSNR (dB)"); a.legend(frameon=False, fontsize=6, loc="center right")
    recessive(a); panel(a, "a")
    a = axs[1]; tds = sorted(TM[TM.model == "U-Net"].train_dose.unique(), key=lambda s: -int(s))
    mc_ = TM[TM.source == "mc"]; XD = {"10": 0, "25": 1, "50": 2}
    if len(tds) == 1:
        # a single training dose: PSNR by test dose instead of a heat map, U-Net and its input
        for name, td, sty in (("OS-SART input", "", dict(ls="-", marker="o", color=MUTED)), (f"U-Net ({DLAB[tds[0]]})", tds[0], dict(ls="-", marker="*", color=INK))):
            v = mc_[(mc_.model == ("OS-SART" if td == "" else "U-Net")) & (mc_.train_dose == td)].groupby("dose").psnr.agg(["mean", "std"]).reindex(["10", "25", "50"])
            a.errorbar(range(3), v["mean"], yerr=v["std"], ms=4, capsize=1.5, lw=1.0, label=name, **sty)
        a.set_xticks(range(3)); a.set_xticklabels([DLAB[k] for k in ("10", "25", "50")]); a.set_xlabel("test dose")
        a.set_ylabel("test PSNR vs $x_\\mathrm{ref}$ (dB)"); a.legend(frameon=False, fontsize=6, loc="lower right"); recessive(a)
    else:
        H = np.array([[mc_[(mc_.model == "U-Net") & (mc_.train_dose == td) & (mc_.dose == k)].psnr.mean() for k in DOSES] for td in tds])
        a.imshow(H, cmap="viridis", aspect="auto")
        for i in range(H.shape[0]):
            for j in range(H.shape[1]): a.text(j, i, f"{H[i, j]:.1f}", ha="center", va="center", fontsize=6, color="white" if H[i, j] < np.nanmean(H) else "black")
        a.set_xticks(range(3)); a.set_xticklabels([DLAB[k] for k in DOSES]); a.set_yticks(range(len(tds))); a.set_yticklabels([DLAB[t] for t in tds])
        a.set_xlabel("test dose"); a.set_ylabel("train dose"); a.spines[:].set_visible(False)
    panel(a, "b")
    a = axs[2]; a.axis("off")
    if EX is not None:
        NUM["ExPatient"] = str(EX["patient"]); NUM["ExDensity"] = str(EX["density"])
        m = EX["mask"].astype(bool); g = EX["sart_full"].astype(np.float32); lo, hi = win_recon(g, m); R0, R1, C0, C1 = bbox(m)
        td = str(tds[0]); keys = [("sart_full", "$x_\\mathrm{ref}$")] + [(f"input_{k}", f"input, {DLAB[k]}") for k in DOSES] + [(f"unet{td}_{k}", f"U-Net, {DLAB[k]}") for k in DOSES]
        sub = a.get_subplotspec().subgridspec(2, 4, wspace=0.03, hspace=0.32)
        for n_, (kk, t) in enumerate(keys):
            r_, c_ = (0, 0) if n_ == 0 else ((0, n_) if n_ <= 3 else (1, n_ - 3))
            ax = fig.add_subplot(sub[r_, c_]); ax.imshow(np.where(m, EX[kk].astype(np.float32), lo)[R0:R1, C0:C1], cmap="gray", vmin=lo, vmax=hi); noaxis(ax)
            if n_ == 0:
                ax.set_title(t, fontsize=6, pad=2); scalebar(ax, PIX, R1 - R0, C1 - C0); tag(ax, "c", x=-0.15, y=1.2)
            else: ax.set_title(("input" if n_ <= 3 else "U-Net") + "\n" + DLAB[DOSES[(n_ - 1) % 3]], fontsize=6, pad=2, linespacing=1.0)
    save(fig, "figS1_training"); plt.show()

# =================================================================== Fig S2 -- calcification gallery
if need("figS2_calcifications", PAN):
    rowsP = [dn for dn in DENS if dn in PAN]
    fig, axs = plt.subplots(len(rowsP), 4, figsize=(W1 * 1.25, 1.1 * len(rowsP)))
    axs = np.atleast_2d(axs)
    for row, dn in zip(axs, rowsP):
        p = PAN[dn]; gc = p["gt_calc"].astype(np.float32); lo, hi = np.percentile(gc[gc > 0], [2, 99.5])
        for ax, (kk, t) in zip(row, [("gt_calc", "$x_\\mathrm{ref}$")] + [(f"calc_OSSART_{k}", f"OS-SART {DLAB[k]}") for k in DOSES]):
            ax.imshow(p[kk], cmap="gray", vmin=lo, vmax=hi); noaxis(ax)
            if dn == rowsP[0]: ax.set_title(t, fontsize=6.3)
        row[0].set_ylabel(DNAME[dn], fontsize=6.3); row[0].yaxis.set_visible(True); scalebar(row[0], PIX, *gc.shape, mm=5)
    plt.tight_layout(w_pad=0.1, h_pad=0.2); save(fig, "figS2_calcifications"); plt.show()

# =================================================================== Fig S3 -- victre_fbp vs sart_full
if need("figS3_victre_scatter", PAN01, B):
    fig, axs = plt.subplots(1, 4, figsize=(W2, 1.9))
    vc = B[(B.method == "GT") & (B.ref == "victre_fbp")]
    for a, dn in zip(axs, DENS):
        p = PAN01.get(dn)
        if p is None or not bool(p["victre_fbp_present"]):
            a.text(0.5, 0.5, "not released", ha="center", va="center", transform=a.transAxes, color=MUTED); a.set_title(DNAME[dn]); continue
        m = p["mask"].astype(bool); x = p["sart_full"][m].astype(np.float32); y = p["victre_fbp"][m].astype(np.float32)
        a.hexbin(x, y, gridsize=45, cmap="Greys", mincnt=1, bins="log", linewidths=0)
        cc = vc[vc.density == dn]["corr"]
        a.set_title(f"{DNAME[dn]}: corr {cc.mean():.2f} (n={len(cc)})" if len(cc) else DNAME[dn], fontsize=6.5)
        a.set_xlabel("$x_\\mathrm{ref}$ (mm$^{-1}$)")
        if dn == DENS[0]: a.set_ylabel("victre_fbp")
    plt.tight_layout(w_pad=0.6); save(fig, "figS3_victre_scatter"); plt.show()

# ------------------------------------------------------------ extra numbers (used in the text)
if B is not None:
    n_all = B[B.ref == "sart_full"].patient.nunique(); n_vf = B[B.ref == "victre_fbp"].patient.nunique()
    NUM["NtestUsed"] = tex_int(n_all); NUM["NtestVFBP"] = tex_int(n_vf)
    for k in DOSE_ALL:
        for mm in ("SART", "FBP"):
            g = B[(B.method == mm) & (B.dose == k) & (B.ref == "sart_full")]
            if len(g): NUM[f"PSNR{mm}{DOSEW[k]}Base"] = f"{g.psnr.mean():.2f}"
    gv = B[(B.method == "GT") & (B.ref == "victre_fbp")]
    if len(gv):
        NUM["CorrVFBPRef"] = f"{gv['corr'].mean():.3f}"; NUM["CorrVFBPMin"] = f"{gv['corr'].min():.3f}"
        for dn in DENS:
            v = gv[gv.density == dn]["corr"]
            if len(v): NUM[f"CorrVFBP{dn.capitalize()}"] = f"{v.mean():.3f}"
    # geometry check: FBP against FBP (full-dose FBP vs VICTRE's FBP)
    ff = B[(B.method == "FBP") & (B.dose == "100") & (B.ref == "victre_fbp")]
    if len(ff):
        NUM["CorrFBPvsVFBP"] = f"{ff['corr'].mean():.3f}"; NUM["CorrFBPvsVFBPsd"] = f"{ff['corr'].std():.3f}"
        NUM["CorrFBPvsVFBPMin"] = f"{ff['corr'].min():.3f}"
        for dn in DENS:
            v = ff[ff.density == dn]["corr"]
            if len(v): NUM[f"CorrFBPvsVFBP{dn.capitalize()}"] = f"{v.mean():.3f}"
        print(f"FBP vs victre_fbp (full dose, n={len(ff)}): {ff['corr'].mean():.3f} +- {ff['corr'].std():.3f}, min {ff['corr'].min():.3f}")
# z alignment: tissue cross-correlation (06), not the per-lesion peak estimate (it sticks to the window edge, 05 B).
if ZS is not None:
    for dn in DENS + ["all"]:
        v = ZS[~ZS.edge.astype(bool)] if dn == "all" else ZS[(ZS.density == dn) & ~ZS.edge.astype(bool)]
        if not len(v): continue
        T_ = dn.capitalize() if dn != "all" else "All"
        NUM[f"Zshift{T_}"] = f"{v.k.mean():+.3f}"; NUM[f"ZshiftSD{T_}"] = f"{v.k.std():.3f}"; NUM[f"ZshiftN{T_}"] = tex_int(len(v))
        for t_, w in enumerate(("Shallow", "Middle", "Deep")):
            if f"k_t{t_}" in v: NUM[f"Zshift{w}{T_}"] = f"{v[f'k_t{t_}'].mean():+.3f}"
        print(f"z (tissue) {dn:<10} n={len(v):>4}  k {v.k.mean():+.3f} +- {v.k.std():.3f} slices")
if PATCH is not None:
    NUM["NPatched"] = tex_int(len(PATCH))
    for dn in ("dense", "hetero", "scattered"):
        v = PATCH[PATCH.density == dn]
        if not len(v): continue
        T_ = dn.capitalize(); NUM[f"DeltaOld{T_}"] = f"{PATCH_DELTA_OLD[dn]:.3f}"
        NUM[f"ZshiftOld{T_}"] = f"{v.k_old.mean():+.3f}"; NUM[f"ZshiftPatch{T_}"] = f"{v.k_new.mean():+.3f}"
        NUM[f"MaskRatio{T_}"] = f"{v.mask_ratio.mean():.4f}"
if DSC is not None:
    for _, r in DSC.iterrows(): NUM[f"DeltaZero{str(r.density).capitalize()}"] = f"{r.delta_zero:.2f}"
# position of the mass peak relative to the VICTRE .loc depth, stacked profile (05 B); in both volumes. Dense is noisy (SE ~ 2.5).
if ZST is not None:
    for _, r in ZST[ZST.kind == "mass"].iterrows():
        w = {"sart_full": "Ref", "victre_fbp": "VFBP"}.get(r.volume)
        if w: NUM[f"LocOffset{w}{str(r.density).capitalize()}"] = f"{r.stacked_offset:+.2f}"; NUM[f"LocOffsetSE{w}{str(r.density).capitalize()}"] = f"{r.stacked_se:.2f}"
# s ~ 1.2-1.3: the cropping of the depth margin (8 + 8 slices); uncropped s ~ 1.00 (05 D)
if SCT is not None:
    for case, w in (("sart_full", "Stored"), ("ext10", "Uncropped"), ("mass_in_margin", "MarginMass")):
        for dn in DENS:
            v = SCT[(SCT.case == case) & (SCT.density == dn)].scale
            if len(v): NUM[f"Scale{w}{dn.capitalize()}"] = f"{v.median():.3f}"
    NUM["NScaleTest"] = str(SCT.patient.nunique())
    v = SCT[SCT.case == "crop_vs_sartfull_maxabs"].scale
    if len(v): NUM["ScaleCropCheck"] = f"{v.max():.1e}"
# low x_ref voxels at the mask edge (06 E)
if MED is not None:
    NUM["NMaskEdge"] = str(MED.patient.nunique())
    for st, w in (("low", "Low"), ("all", "All")):
        for dn in DENS:
            v = MED[(MED.set == st) & (MED.density == dn)]
            if len(v):
                NUM[f"MaskEdge{w}{dn.capitalize()}"] = f"{100 * v.frac_d2.mean():.0f}"
                NUM[f"MaskZedge{w}{dn.capitalize()}"] = f"{100 * v.frac_zedge.mean():.0f}"
if DF is not None:
    for dn in DENS:
        v = DF[(DF.volume == "sart_full") & (DF.density == dn)].datafit
        if len(v): NUM[f"Datafit{dn.capitalize()}"] = f"{v.median():.4f}"
""")

md(r"""
## F · `numbers.tex` and summary

**Every number** in the text comes from here. If a new number is needed it is added to `NUM` and the notebook is run again.
""")
code(r"""
if K is not None:
    NUM["Npatients"] = tex_int(len(K)); NUM["Nslices"] = tex_int(K.slices.sum())
    for s in ("train", "val", "test"): NUM[f"N{s.capitalize()}"] = tex_int((K.split == s).sum())
    for dn in DENS:
        R = K[K.density == dn]
        NUM[f"N{dn.capitalize()}"] = tex_int(len(R)); NUM[f"NVFBP{dn.capitalize()}"] = tex_int(R.victre_fbp.sum())
    NUM["NSignalPresent"] = tex_int(K.signal_present.sum()); NUM["NSignalAbsent"] = tex_int((1 - K.signal_present).sum())
    NUM["NMasses"] = tex_int(K.n_mass.sum()); NUM["NCalcs"] = tex_int(K.n_calc.sum()); NUM["NControls"] = tex_int(K.n_control.sum())
    NUM["NVFBP"] = tex_int(K.victre_fbp.sum()); NUM["PctVFBP"] = f"{100 * K.victre_fbp.mean():.1f}"
    NUM["SizeGB"] = f"{K.size_mb.sum() / 1000:.1f}"; NUM["MBperPatient"] = f"{K.size_mb.mean():.1f}"
    NUM["SatMax"] = f"{K.victre_fbp_saturated.max():.1e}"
    NUM["NSPnoLesion"] = str(int(((K.signal_present == 1) & (K.n_mass + K.n_calc == 0)).sum()))
for dn in DENS: NUM[f"Slices{dn.capitalize()}"] = str(SLC[dn])
# config.json: all numerical parameters of the generator (Methods are written from here)
for k, v in CFGJ.items():
    name = "Cfg" + "".join(w.capitalize() for w in k.split("_"))
    if isinstance(v, bool): continue
    if isinstance(v, (int, float)): NUM[name] = f"{v:g}"
    elif isinstance(v, (list, tuple)) and v and all(isinstance(x, (int, float)) for x in v):
        for i, x in enumerate(v): NUM[name + "ABCDEFGH"[i]] = f"{x:g}"
NUM["CfgNoiseDN"] = f"{CFGJ.get('electronic_noise_electrons', 5200) * CFGJ.get('dn_per_electron', 8 * 0.001469):.0f}"
NUM["DmPixelMM"] = f"{float(CFGJ.get('native_pixel_mm', 0.085)) * int(CFGJ.get('dm_downsample', 2)):.2f}"
NUM["NativeDetRows"], NUM["NativeDetCols"] = tex_int(3000), tex_int(1500)          # VICTRE detector, 0.085 mm (config comment)
# Protocol constants: the settings of the evaluation and training notebooks (03, 04, 06); in one place, the text uses these.
PROTO = dict(UNetTrainPerDensity=150, UNetValPerDensity=10, UNetSlicesPerPatient=16, UNetSteps=30000,
             UNetBatch=8, UNetCrop=256, UNetChannels=32, UNetDepth=4, NPSPatch=64, SDNRCoreHalf=4, SDNRBgHalf=12,
             LowThr=0.02, ZshiftKmax=4, ZshiftSigma=4, NPilot=384, DeltaScanN=6, NTciaSubjects=2994,
             SDNRCoreW=9, SDNRCoreZ=3, SDNRBgW=25, MassCropMM=40, CalcCropMM=20, AxisBreakMM=130,
             DmRows=1500, DmCols=750, NTuneVal=4, NDeltaValues=4, NPilotFatty=30, UNetLR=0.001, UNetWD=0.0001,
             FootprintDilatePx=10, ValQuarterOurs=0.113, ValQuarterMC=0.118)
# ValQuarter*: during development of the method, a comparison with a real MC-GPU quarter-dose simulation of one
# phantom (FBP noise s.d.), from the development release's "Noise Model ... Checks". Not in the dataset.
# The one TCIA record not in the dataset: its VICTRE reconstruction has 5 slices instead of 62 (TCIA NBIA getSeries).
NUM["ExcludedSubject"], NUM["ExcludedReconSlices"] = "208084664", "5"
NUM["DeltaScanLo"], NUM["DeltaScanHi"] = "19.0", "20.5"            # ends of D_GRID in 06
for k, v in PROTO.items(): NUM[k] = tex_int(v) if isinstance(v, int) and v >= 1000 else f"{v:g}"
NUM["UNetTrainPatients"] = tex_int(4 * PROTO["UNetTrainPerDensity"]); NUM["UNetValPatients"] = str(4 * PROTO["UNetValPerDensity"])
if VAL:
    mt = re.search(r"Files checked: (\d+)\. Problems: (\d+)", VAL)
    if mt: NUM["NValidated"], NUM["NProblems"] = tex_int(mt.group(1)), mt.group(2)
    mt = re.search(r"corr\(victre_fbp, sart_full\)[^|]*\|\s*([\d.]+) \[([\d.]+), ([\d.]+)\]", VAL)
    if mt: NUM["CorrVFBPRefLo"], NUM["CorrVFBPRefHi"] = mt.group(2), mt.group(3)
if K is not None:
    NUM["SigmaEMin"], NUM["SigmaEMax"] = f"{K.sigma_e.min():.1f}", f"{K.sigma_e.max():.1f}"
    NUM["SatMean"] = f"{K[K.victre_fbp == 1].victre_fbp_saturated.mean():.1e}"
    NUM["NFattyNoVFBP"] = tex_int(((K.density == "fatty") & (K.victre_fbp == 0)).sum())
    for s_ in ("train", "val", "test"):
        R = K[K.split == s_]; NUM[f"N{s_.capitalize()}SP"] = tex_int(R.signal_present.sum())
    NUM["NSPFewMasses"] = str(int(((K.signal_present == 1) & (K.n_mass < 4)).sum()))
    NUM["NSPFewCalcs"] = str(int(((K.signal_present == 1) & (K.n_calc < 4)).sum()))
    NUM["NSANoControl"] = str(int(((K.signal_present == 0) & (K.n_control == 0)).sum()))
    NUM["MBMin"], NUM["MBMax"] = f"{K.size_mb.min():.0f}", f"{K.size_mb.max():.0f}"
# check of 07: rebuild from the 16-bit proj_full with the old DELTA = the recorded sart_full (built from 32-bit)
_yd = DATA/"delta_patch_audit.json"
if _yd.exists():
    _j = json.loads(_yd.read_text())
    NUM["PatchCheckMaxRel"] = f"{100 * max(v['max_rel_p995'] for v in _j.values()):.2f}"
    NUM["PatchCheckN"] = str(len(_j))
    NUM["PatchCheckMeanRel"] = f"{100 * max(v['mean_abs_rel'] for v in _j.values()):.3f}"
    NUM["PatchCheckMaskAgree"] = f"{100 * min(v['mask_agree'] for v in _j.values()):.4f}"
if TC is not None: NUM["UNetValPSNRInput"] = f"{TC.val_psnr_input.iloc[-1]:.2f}"; NUM["UNetValPSNRFinal"] = f"{TC.val_psnr.iloc[-1]:.2f}"
NUM["PixelMM"] = f"{PIX:.2f}"
NUM["OffxC"] = f"{OFFX_C:+.3f}"
for dn in DENS: NUM[f"NativeRows{dn.capitalize()}"] = tex_int(NATR[dn])
NUM.update({f"Delta{dn.capitalize()}": f"{DELTA[dn]:.3f}" for dn in DENS})
NUM.update({"DeltaPublished": f"{DELTA_PUB:.3f}", "DeltaFattyOld": f"{PILOT_FATTY['delta']:.3f}",
            "ZoffFattyOld": f"{PILOT_FATTY['gt'][0]:+.2f}", "ZseFattyOld": f"{PILOT_FATTY['gt'][1]:.2f}",
            "NPilotFattyMasses": str(PILOT_FATTY['gt'][2])})

bad = [k for k in NUM if not re.fullmatch(r"[A-Za-z]+", k)]
assert not bad, f"LaTeX macro names must contain letters only: {bad}"
L = ["% GENERATED AUTOMATICALLY -- do not edit. Produced by: notebooks/02_figures_tables_colab.ipynb", ""]
def texfmt(v):
    v = str(v)
    m_ = re.fullmatch(r"([+-]?)(\d+(?:\.\d+)?)e([+-])0*(\d+)", v)
    if m_: return "\\ensuremath{" + ("-" if m_.group(1) == "-" else "") + m_.group(2) + "\\times10^{" + ("-" if m_.group(3) == "-" else "") + m_.group(4) + "}}"
    if re.fullmatch(r"-[\d.,{}]+", v): return "\\ensuremath{-}" + v[1:]
    return v
NUM.update({"DosePctA": "50", "DosePctB": "25", "DosePctC": "10"})
L += [f"\\newcommand{{\\{k}}}{{{texfmt(v)}}}" for k, v in sorted(NUM.items())]
(RES/"numbers.tex").write_text("\n".join(L) + "\n")
print(f"numbers.tex: {len(NUM)} macros\n")
print("produced:", ", ".join(MADE))
print("skipped :", ", ".join(SKIPPED) or "none")
print(f"\nfor Overleaf: {TAB}/*.tex, {FIG}/*.pdf, {RES}/numbers.tex; for Fig 1 {THUMB}/*.png")
""")

nb = {"cells": [], "metadata": {"colab": {"provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
for kind, src in cells:
    lines = src.split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    c = {"cell_type": kind, "metadata": {}, "source": source}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    nb["cells"].append(c)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("written", OUT, len(cells), "cells")
