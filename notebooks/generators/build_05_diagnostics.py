"""Generates 05_diagnostics_colab.ipynb. Answers four measurement questions that came up when reviewing the output of 02."""
import json, sys
from pathlib import Path

OUT = Path(sys.argv[1])
cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md(r"""
# VICTRE-Paired v11 — diagnostics (Colab, GPU, ~1 h)

Four questions that came up when reviewing the first output of 02. None changes the dataset; all decide
**what the paper states**. Outputs in `VICTRE-PAIRED-v11_analysis/diagnostics/`, read by 02.

| | cell | question | time |
|---|---|---|---|
| **A** | setup | GPU, ASTRA, Drive | 2 min |
| **B** | settings | paths, helpers | 1 min |
| **C** | z alignment, stacked profile | In Fig 7d the dense/hetero boxes span the whole ±3 slices: does the per-lesion peak estimator stick to the window edge? Robust measure: the peak of the mean profile per density | ~30 min |
| **D** | scale `s = 1.25` | `sart_full` explains ~80 % of the measured attenuation. Is it the depth margin (8+8 slices, cropped after SART), rays that leave the volume, or too few iterations? | ~10 min |
| **E** | noise ratio < 1 | 0.91–0.97 in Supplementary Table S5: is the dose model incomplete, or does the estimator read low? Re-noise with the same model and measure with the same estimator | ~3 min |
| **F** | `x_ref` zeros, two clusters in fatty | Supplementary Fig S3: voxels with `x_ref = 0` inside the mask and a second low cluster in fatty. Distribution by depth | ~5 min |
| **G** | summary | `diagnostics/summary.json` | — |

Every cell resumes where it stopped. Drive reads have a timeout: if one hangs, the cell stops and says what
to do.
""")

md("## A · Setup")
code(r"""
import subprocess, sys
out = subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
assert out, "NO GPU -> Runtime > Change runtime type > GPU"
print("GPU:", out)

!pip -q install astra-toolbox pydicom tqdm scikit-image 2>&1 | tail -2
import astra
print("ASTRA", astra.__version__, "| CUDA:", astra.astra.use_cuda())
assert astra.astra.use_cuda(), "ASTRA has no CUDA build"

from google.colab import drive
drive.mount('/content/drive')
""")

md("## B · SETTINGS")
code(r"""
import os, sys, json, time, collections
from pathlib import Path
import numpy as np, pandas as pd
from scipy import ndimage
import concurrent.futures as _cf

NB_DIR = Path("/content/drive/MyDrive/victre_paired/code")
DATA   = Path("/content/drive/MyDrive/New_DBT/VICTRE-PAIRED-v11")
RES    = Path("/content/drive/MyDrive/New_DBT/VICTRE-PAIRED-v11_analysis")
DIAG   = RES/"diagnostics"

N_S_PER_DEN = 2       # D and E: phantoms per density (test, those with victre_fbp and lesions preferred)
DZ_MAX      = 10      # C: profile window +-10 slices (the estimator of 01 used +-3)
N_BOOT      = 1000

for d in (DIAG, DIAG/"zprof"):
    d.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(NB_DIR)); os.chdir(NB_DIR)
import dataset_generate as G
import dataset_test as T
assert all(v == 19.5 for v in G.DELTA.values()), f"dataset_generate.py on Drive is old (before the DELTA patch): {G.DELTA}"
DENS = list(G.DENSITIES)
files = sorted(p for p in (DATA/"test").glob("*.npz") if ".partial" not in p.name)
K = pd.read_csv(RES/"registry.csv", dtype={"patient": str})
KT = K[K.split == "test"].set_index("patient")
print(f"test: {len(files)} files, registry {len(KT)}")

_LOADER = _cf.ThreadPoolExecutor(1)
def load_full(f, keys=None, timeout=900):
    # A hanging Drive read does not fail, it waits: this one is time-limited and says what to do if it hangs.
    def _rd():
        with np.load(f) as z:
            return {k: z[k] for k in (keys or z.files)}
    try:
        return _LOADER.submit(_rd).result(timeout=timeout)
    except _cf.TimeoutError:
        raise RuntimeError(f"{f.name}: Drive did not respond in {timeout} s. Restart the session, A -> B -> "
                           "this cell (resumes where it stopped).")

# ------------------------------------------------------------------ the same contrast definition as in 01
R_LES = int(round(G.CFG.lesion_radius_mm / (G.CFG.native_pixel_mm * G.CFG.downsample)))
def contrast(sl, r, c):
    yy, xx = np.ogrid[:sl.shape[0], :sl.shape[1]]
    d2 = (yy - r)**2 + (xx - c)**2
    core, ring = d2 <= R_LES**2, (d2 > (2*R_LES)**2) & (d2 <= (3*R_LES)**2)
    if core.sum() < 5 or ring.sum() < 20: return np.nan
    return float(sl[core].mean() - sl[ring].mean())

def profile(vol, z, r, c, dmax=DZ_MAX):
    zi = int(round(z)); S = vol.shape[0]
    return np.array([contrast(vol[zi + d].astype(np.float64), r, c) if 0 <= zi + d < S else np.nan
                     for d in range(-dmax, dmax + 1)])

def peak(y, dz):
    # Peak by parabolic interpolation; if the peak is at the window edge, the edge is returned (with a flag).
    ok = np.isfinite(y)
    if ok.sum() < 3: return np.nan, True
    k = int(np.nanargmax(np.where(ok, y, -np.inf)))
    if k == 0 or k == len(y) - 1 or not (ok[k-1] and ok[k+1]): return float(dz[k]), True
    a, b, c = y[k-1], y[k], y[k+1]; den = a - 2*b + c
    sub = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
    return float(dz[k] + np.clip(sub, -1, 1)), False

# ------------------------------------------------------------------ for D: OS-SART with a depth margin
def os_sart_ext(geo, proj, valid, iterations=None):
    # the SAME as dataset_generate.Geometry.os_sart, with two differences: the number of passes can be chosen and the
    # volume is returned UNCROPPED (S + 2m slices). The cropped version = the sart_full recipe.
    it = G.CFG.sart_iterations if iterations is None else iterations
    vectors = G.cone_vectors()
    vid = astra.data3d.create("-vol", geo.vol_geom_extended, 0)
    subs = []
    for start in range(G.CFG.sart_subsets):
        views = np.arange(start, G.CFG.num_views, G.CFG.sart_subsets)
        pg = astra.create_proj_geom("cone_vec", G.CFG.det_rows, G.CFG.det_cols, vectors[views])
        subs.append((astra.data3d.create("-proj3d", pg, np.ascontiguousarray(proj[views].transpose(1, 0, 2), np.float32)),
                     astra.data3d.create("-proj3d", pg, np.ascontiguousarray(valid[views].transpose(1, 0, 2), np.float32))))
    for _ in range(it):
        for sid, mid in subs:
            cfg = astra.astra_dict("SIRT3D_CUDA")
            cfg["ProjectionDataId"], cfg["ReconstructionDataId"] = sid, vid
            cfg["option"] = {"MinConstraint": 0.0, "SinogramMaskId": mid}
            alg = astra.algorithm.create(cfg); astra.algorithm.run(alg, 1); astra.algorithm.delete(alg)
    v = astra.data3d.get(vid); astra.data3d.delete([vid] + [i for p in subs for i in p])
    return np.ascontiguousarray(v.transpose(0, 2, 1))          # (S+2m, rows, cols)

def project_ext(geo, vol_ext):
    sid, s = astra.create_sino3d_gpu(np.ascontiguousarray(vol_ext.transpose(0, 2, 1), np.float32),
                                     geo.proj_geom, geo.vol_geom_extended)
    astra.data3d.delete(sid); return s.transpose(1, 0, 2)

def crop(geo, vol_ext):
    m = G.CFG.sart_depth_margin_slices
    return vol_ext[m:m + geo.slices]

def pad_ext(geo, vol):
    m = G.CFG.sart_depth_margin_slices
    return np.pad(vol, ((m, m), (0, 0), (0, 0)))

def fit(ax, p, sel):
    # the datafit of 01: s by least squares, residual ||s A x - p|| / ||p||
    a, b = ax[sel].astype(np.float64), p[sel].astype(np.float64)
    s = float(a @ b / max(a @ a, 1e-12))
    return s, float(np.linalg.norm(s * a - b) / max(np.linalg.norm(b), 1e-12))

def pick_patients(n):
    out = []
    for dn in DENS:
        c = KT[(KT.density == dn)].sort_values(["victre_fbp", "signal_present"], ascending=False)
        out += [DATA/"test"/f"{pid}.npz" for pid in c.index[:n]]
    return out
print("ready. R_LES =", R_LES, "px")
""")

md(r"""
## C · z alignment — stacked profile

For each test lesion, the contrast profile over ±10 slices in `sart_full` and (if present) `victre_fbp`
(the `contrast` definition of 01, same core and ring). Then:

1. **Per lesion**, the peak with a ±3 and a ±10 window: how many lesions peak **at the window edge**
   (where the estimator fails). Fig 7d of 01 used ±3.
2. **Stacked:** the profiles are averaged per density (each divided by its own RMS, which does not move
   the peak), and the peak of the mean profile is taken. Uncertainty by phantom bootstrap.
3. **Depth trend:** the same stacked peak, by depth tercile.

One file per phantom in `diagnostics/zprof/<id>.json`; finished phantoms are skipped.
""")
code(r"""
sp_files = [f for f in files if f.stem in KT.index and int(KT.loc[f.stem, "signal_present"]) == 1]
todo = [f for f in sp_files if not (DIAG/"zprof"/f"{f.stem}.json").exists()]
print(f"test phantoms with lesions {len(sp_files)}, {len(sp_files)-len(todo)} finished, {len(todo)} left", flush=True)
t0 = time.time()
for i, f in enumerate(todo, 1):
    print(f"[{i}/{len(todo)}] {f.stem} reading ...", end=" ", flush=True)
    d = load_full(f, ["sart_full", "victre_fbp", "victre_fbp_present", "victre_fbp_scale", "lesions", "density"])
    gt = d["sart_full"].astype(np.float32)
    vf = (d["victre_fbp"].astype(np.float32) / 65535.0 * float(d["victre_fbp_scale"])) if bool(d["victre_fbp_present"]) else None
    rows = []
    for s_, r_, c_, num, kind in d["lesions"]:
        if int(kind) not in (0, 1): continue
        for vn, v in (("sart_full", gt), ("victre_fbp", vf)):
            if v is None: continue
            rows.append(dict(volume=vn, kind=int(kind), num=int(num), s=float(s_), S=int(gt.shape[0]),
                             prof=[None if not np.isfinite(x) else float(x) for x in profile(v, s_, r_, c_)]))
    rec = dict(patient=f.stem, density=str(d["density"]), rows=rows)
    tmp = DIAG/"zprof"/f"{f.stem}.json.tmp"; tmp.write_text(json.dumps(rec)); tmp.rename(DIAG/"zprof"/f"{f.stem}.json")
    el = time.time() - t0
    print(f"{len(rows)} profiles | ~{el/i*(len(todo)-i)/60:.0f} min left", flush=True)

# ---- summary
DZ = np.arange(-DZ_MAX, DZ_MAX + 1)
L = []
for p in sorted((DIAG/"zprof").glob("*.json")):
    r = json.loads(p.read_text())
    for x in r["rows"]:
        y = np.array([np.nan if v is None else v for v in x["prof"]], float)
        L.append(dict(patient=r["patient"], density=r["density"], volume=x["volume"], kind=x["kind"],
                      zfrac=x["s"] / x["S"], y=y))
L = pd.DataFrame(L)
c3 = (np.abs(DZ) <= 3)
def per_lesion(y, win):
    return peak(np.where(win, y, np.nan)[win], DZ[win])
L["off3"], L["edge3"] = zip(*[per_lesion(y, c3) for y in L.y])
L["off10"], L["edge10"] = zip(*[per_lesion(y, np.ones_like(c3)) for y in L.y])

def stacked(sub):
    Y = np.stack([y / np.sqrt(np.nanmean(y**2)) for y in sub.y if np.isfinite(y).sum() >= 5 and np.nanmean(y**2) > 0])
    return peak(np.nanmean(Y, 0), DZ)[0], np.nanmean(Y, 0)

rng = np.random.default_rng(0)
def boot(sub):
    pats = sub.patient.unique(); g = {p: sub[sub.patient == p] for p in pats}
    est = stacked(sub)[0]
    bs = [stacked(pd.concat([g[p] for p in rng.choice(pats, len(pats))]))[0] for _ in range(N_BOOT)]
    return est, float(np.nanstd(bs, ddof=1))

OZ, PROF, TER = [], [], []
for vn in ("sart_full", "victre_fbp"):
    for kind, kn in ((1, "mass"), (0, "calc")):
        for dn in DENS + ["all"]:
            sub = L[(L.volume == vn) & (L.kind == kind) & ((L.density == dn) if dn != "all" else True)]
            if sub.patient.nunique() < 3: continue
            est, se = boot(sub)
            OZ.append(dict(volume=vn, kind=kn, density=dn, n=len(sub), n_pat=sub.patient.nunique(),
                           stacked_offset=est, stacked_se=se,
                           edge3=float(sub.edge3.mean()), edge10=float(sub.edge10.mean()),
                           mean_off3=float(sub.off3.mean()), mean_off10_interior=float(sub[~sub.edge10].off10.mean())))
            prof = stacked(sub)[1]
            PROF += [dict(volume=vn, kind=kn, density=dn, dz=int(z), y=float(v)) for z, v in zip(DZ, prof)]
            if kind == 1:
                q = pd.qcut(sub.zfrac, 3, labels=False, duplicates="drop")
                for t in sorted(q.unique()):
                    st = sub[q == t]
                    if st.patient.nunique() < 3: continue
                    e_, s_ = boot(st)
                    TER.append(dict(volume=vn, density=dn, tercile=int(t), zfrac=float(st.zfrac.mean()),
                                    S=float(np.mean([KT.loc[p, "slices"] for p in st.patient])), n=len(st),
                                    stacked_offset=e_, stacked_se=s_))
OZ, PROF, TER = pd.DataFrame(OZ), pd.DataFrame(PROF), pd.DataFrame(TER)
OZ.to_csv(DIAG/"zalign_stacked.csv", index=False); PROF.to_csv(DIAG/"zalign_profiles.csv", index=False)
TER.to_csv(DIAG/"zalign_tercile.csv", index=False)
pd.set_option("display.width", 200)
print("\n--- masses and calcifications: stacked peak (slices) +- phantom-bootstrap s.e.; edge fraction ---")
print(OZ.round(3).to_string(index=False))
print("\n--- masses, stacked peak by depth tercile ---")
print(TER.round(3).to_string(index=False))
""")

md(r"""
## D · Scale `s` — which component?

In 2 test phantoms per density (`N_S_PER_DEN`), `s` and the residual `‖s·A·x − p‖/‖p‖` against the full-dose
Monte Carlo projections, on valid rays:

| row | volume | forward model |
|---|---|---|
| `sart_full` | the recorded reference (cropped) | record grid |
| `ext10` | the same recipe, **uncropped** (S + 16 slices) | extended grid |
| `ext10_crop` | `ext10` cropped (= `sart_full`, a check) | record grid |
| `ext30` | 3× the passes, uncropped | extended grid |

Also the **volume coverage:** the fraction of valid rays in the breast shadow that never enter the record grid,
and their share of `‖p‖²`, per view.
""")
code(r"""
SF = pick_patients(N_S_PER_DEN)
SROWS, COV = [], []
for i, f in enumerate(SF, 1):
    t = time.time()
    d = load_full(f, ["sart_full", "proj_full", "proj_valid", "density"])
    dn, gt = str(d["density"]), d["sart_full"].astype(np.float32)
    geo = G.Geometry(dn, gt.shape[0])
    p = d["proj_full"].astype(np.float32); valid = d["proj_valid"].astype(bool); sel = valid & (p > 0)
    e10 = os_sart_ext(geo, p, valid); e30 = os_sart_ext(geo, p, valid, 3 * G.CFG.sart_iterations)
    cases = {"sart_full": geo.project(gt), "ext10": project_ext(geo, e10),
             "ext10_crop": geo.project(np.ascontiguousarray(crop(geo, e10))), "ext30": project_ext(geo, e30)}
    for name, ax in cases.items():
        s, r = fit(ax, p, sel)
        SROWS.append(dict(patient=f.stem, density=dn, case=name, scale=s, res=r))
    m = G.CFG.sart_depth_margin_slices
    SROWS.append(dict(patient=f.stem, density=dn, case="mass_in_margin",
                      scale=float(np.r_[e10[:m].ravel(), e10[-m:].ravel()].sum() / max(e10.sum(), 1e-12)), res=np.nan))
    SROWS.append(dict(patient=f.stem, density=dn, case="crop_vs_sartfull_maxabs",
                      scale=float(np.abs(crop(geo, e10) - gt).max()), res=np.nan))
    L1 = geo.project(np.ones_like(gt)); L1e = project_ext(geo, np.ones((gt.shape[0] + 2*m,) + gt.shape[1:], np.float32))
    for v in range(p.shape[0]):
        sv = sel[v]; e2 = float((p[v][sv]**2).sum())
        for grid, Lg in (("record", L1), ("extended", L1e)):
            out_ = sv & (Lg[v] < 1e-3)
            COV.append(dict(patient=f.stem, density=dn, view=v, grid=grid, frac_rays=float(out_.sum() / max(sv.sum(), 1)),
                            frac_energy=float((p[v][out_]**2).sum() / max(e2, 1e-12))))
    print(f"[{i}/{len(SF)}] {f.stem} {dn:<9} {time.time()-t:.0f} s  | " +
          "  ".join(f"{r_['case']} s={r_['scale']:.3f} r={r_['res']:.4f}" for r_ in SROWS[-6:-2]), flush=True)
SR, CV = pd.DataFrame(SROWS), pd.DataFrame(COV)
SR.to_csv(DIAG/"scale_test.csv", index=False); CV.to_csv(DIAG/"coverage.csv", index=False)
print("\n--- s and residual, median ---")
print(SR[SR.case.isin(["sart_full", "ext10", "ext10_crop", "ext30"])].groupby(["case", "density"])[["scale", "res"]].median().round(4).unstack())
print("\nfraction of the extended volume's mass in the depth margin (median):",
      SR[SR.case == "mass_in_margin"].groupby("density").scale.median().round(3).to_dict())
print("check, cropped ext10 = sart_full (max |diff|):", SR[SR.case == "crop_vs_sartfull_maxabs"].scale.max())
print("\n--- rays that do not enter the grid (valid, in the breast shadow): fraction and share of ||p||^2, end and centre views ---")
print(CV[CV.view.isin([0, 12, 24])].groupby(["grid", "view"])[["frac_rays", "frac_energy"]].median().round(4))
""")

md(r"""
## E · Noise ratio < 1 — the model or the estimator?

On the same phantoms, the dataset's **own** `reduce_dose` is applied to the file's `proj_full` with a new seed
and measured with `dataset_test.noise_ratios`. If the ratio is again ~0.91–0.97, the shortfall is a property of
the estimator (or of `float16` storage), not of the recorded noise. Second row:
the same without rounding to `float16`.
""")
code(r"""
NR = []
for i, f in enumerate(SF, 1):
    d = load_full(f)
    stored = T.noise_ratios(d)
    rng = np.random.default_rng(12345)
    for tag, cast in (("regen_f16", np.float16), ("regen_f32", np.float32)):
        d2 = dict(d)
        for prefix, full, n0k, sek in (("proj", d["proj_full"].astype(np.float32), "photons_per_pixel", "electronic_sigma"),
                                       ("dm", d["dm_full"].astype(np.float32), "dm_photons_per_pixel", "dm_electronic_sigma")):
            fp = full > 0
            for fr in G.CFG.dose_fractions:
                d2[G.dose_key(prefix, fr)] = G.reduce_dose(full, fp, float(d[n0k]), float(d[sek]), fr, rng).astype(cast)
        for k, v in T.noise_ratios(d2).items():
            NR.append(dict(patient=f.stem, density=str(d["density"]), key=k, source=tag, ratio=v))
    for k, v in stored.items():
        NR.append(dict(patient=f.stem, density=str(d["density"]), key=k, source="stored", ratio=v))
    print(f"[{i}/{len(SF)}] {f.stem}", flush=True)
NR = pd.DataFrame(NR); NR.to_csv(DIAG/"noise_regen.csv", index=False)
print(NR.groupby(["key", "source"]).ratio.mean().unstack().round(3))
""")

md(r"""
## F · `x_ref` zeros and the second cluster in fatty

The fraction of voxels with `sart_full == 0` inside the mask and its distribution over slices; the share of the
low cluster (`x_ref < 0.02 mm⁻¹`) in fatty by slice. The source of the two images in Supplementary Fig S3.
""")
code(r"""
XR = []
for i, f in enumerate(SF, 1):
    d = load_full(f, ["sart_full", "breast_mask", "density"])
    gt, m, dn = d["sart_full"].astype(np.float32), d["breast_mask"].astype(bool), str(d["density"])
    S = gt.shape[0]
    for z in range(S):
        mz = m[z]
        if not mz.any(): continue
        XR.append(dict(patient=f.stem, density=dn, z=z, zfrac=(z + 0.5) / S, n_mask=int(mz.sum()),
                       frac_zero=float((gt[z][mz] == 0).mean()), frac_low=float((gt[z][mz] < 0.02).mean()),
                       median=float(np.median(gt[z][mz]))))
    print(f"[{i}/{len(SF)}] {f.stem}", flush=True)
XR = pd.DataFrame(XR); XR.to_csv(DIAG/"xref_depth.csv", index=False)
w = lambda g, c: float(np.average(g[c], weights=g.n_mask))
print("fraction of x_ref = 0 inside the mask:", {dn: round(w(g, "frac_zero"), 4) for dn, g in XR.groupby("density")})
print("fraction of x_ref < 0.02 inside the mask:", {dn: round(w(g, "frac_low"), 4) for dn, g in XR.groupby("density")})
XR["zbin"] = pd.cut(XR.zfrac, [0, .1, .25, .5, .75, .9, 1.0])
print("\nlow cluster by depth (fraction):")
XR["w_low"] = XR.frac_low * XR.n_mask
t_ = XR.groupby(["density", "zbin"], observed=True)[["w_low", "n_mask"]].sum()
print((t_.w_low / t_.n_mask).unstack().round(3).to_string())
""")

md("## G · Summary → `diagnostics/summary.json`")
code(r"""
oz = {}
OZ = pd.read_csv(DIAG/"zalign_stacked.csv"); TER = pd.read_csv(DIAG/"zalign_tercile.csv")
SR = pd.read_csv(DIAG/"scale_test.csv"); CV = pd.read_csv(DIAG/"coverage.csv")
NR = pd.read_csv(DIAG/"noise_regen.csv"); XR = pd.read_csv(DIAG/"xref_depth.csv")
for r in OZ.itertuples():
    oz[f"z_{r.volume}_{r.kind}_{r.density}"] = dict(offset=r.stacked_offset, se=r.stacked_se, n=r.n,
                                                    edge3=r.edge3, edge10=r.edge10)
for case in ("sart_full", "ext10", "ext30"):
    q = SR[SR.case == case]; oz[f"scale_{case}"] = dict(scale=float(q.scale.median()), res=float(q.res.median()))
oz["mass_in_margin"] = float(SR[SR.case == "mass_in_margin"].scale.median())
for g, q in CV.groupby("grid"):
    oz[f"cov_{g}"] = dict(frac_rays_edge=float(q[q.view.isin([0, 24])].frac_rays.median()),
                          frac_energy_edge=float(q[q.view.isin([0, 24])].frac_energy.median()),
                          frac_energy_centre=float(q[q.view == 12].frac_energy.median()))
oz["noise"] = NR.groupby(["key", "source"]).ratio.mean().unstack().round(4).to_dict()
oz["xref_zero"] = {dn: float(np.average(g.frac_zero, weights=g.n_mask)) for dn, g in XR.groupby("density")}
oz["xref_low"] = {dn: float(np.average(g.frac_low, weights=g.n_mask)) for dn, g in XR.groupby("density")}
(DIAG/"summary.json").write_text(json.dumps(oz, indent=1, default=float))
print(json.dumps(oz, indent=1, default=float)[:3000])
""")

nb = {"cells": [], "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
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
