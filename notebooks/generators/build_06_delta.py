"""Generates 06_delta_colab.ipynb. Checks the depth-alignment finding of 05: is DELTA 19.5 for dense/hetero/scattered too?"""
import json, sys
from pathlib import Path

OUT = Path(sys.argv[1])
cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md(r"""
# VICTRE-Paired v11 — DELTA check (Colab, GPU, ~45 min)

**Question.** The stacked z profile of 05 found `sart_full` shifted by −0.57 ± 0.15 slices in hetero and −0.48 ± 0.09
in scattered; VICTRE's own volume (`victre_fbp`) gives +0.30 / +0.16 on the same lesions.
In fatty, after the correction of DELTA 20.807 → 19.5, the offset is +0.26. The DELTA of the other three classes
comes from v9 (19.765 / 20.242 / 20.091). Moved to 19.5, their offsets would appear to coincide with
VICTRE's. This notebook tests this with two **lesion-independent** measurements.

| | cell | what it measures | time |
|---|---|---|---|
| **A** | setup | GPU, ASTRA, Drive | 2 min |
| **B** | settings | paths, helpers | 1 min |
| **C** | tissue z offset | z offset between `sart_full` and `victre_fbp` in all test phantoms, by cross-correlation of **all tissue** (needs no lesion, works in dense too). Also by depth third | ~20 min |
| **D** | DELTA scan | 6 phantoms per density, OS-SART rebuilt with DELTA = 19.0 / 19.5 / 20.0 / 20.5. For each, the offset against `victre_fbp` → the DELTA that zeroes the offset. Also the lesion offset and the data fit | ~20 min |
| **E** | mask edge | whether the low `x_ref` voxels of 05 F lie on the in-plane edge of the mask or at the depth ends | ~2 min |
| **F** | decision | `delta/summary.json` and a plain-language result | — |

**Expected (if 19.5 is right):** in C an offset of hetero ≈ −0.9, scattered ≈ −0.6 slices; in D
the DELTA that zeroes the offset ≈ 19.5 in every class. Does not change the dataset; output in `analysis/delta/`.
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
import os, sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import ndimage
import concurrent.futures as _cf

NB_DIR = Path("/content/drive/MyDrive/victre_paired/code")
DATA   = Path("/content/drive/MyDrive/victre_paired/data")
RES    = Path("/content/drive/MyDrive/victre_paired/analysis")
OUTD   = RES/"delta"

N_D      = 6                          # D: phantoms per density
D_GRID   = [19.0, 19.5, 20.0, 20.5]   # D: DELTA values tried (mm)
KMAX     = 4                          # search window of the z offset (slices)
HP_SIGMA = 4.0                        # high-pass filter of the tissue (pixels, in plane)
EDGE_Z   = 2                          # end slices skipped in the correlation

OUTD.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(NB_DIR)); os.chdir(NB_DIR)
import dataset_generate as G
assert G.DELTA["fatty"] == 19.5, f"dataset_generate.py on Drive is old: {G.DELTA}"
DELTA0 = dict(G.DELTA)                # the values in the record
DENS = list(G.DENSITIES)
files = sorted(p for p in (DATA/"test").glob("*.npz") if ".partial" not in p.name)
K = pd.read_csv(RES/"registry.csv", dtype={"patient": str})
KT = K[K.split == "test"].set_index("patient")
print("DELTA in the record:", DELTA0, "| test:", len(files))

_LOADER = _cf.ThreadPoolExecutor(1)
def load_full(f, keys=None, timeout=900):
    def _rd():
        with np.load(f) as z:
            return {k: z[k] for k in (keys or z.files)}
    try:
        return _LOADER.submit(_rd).result(timeout=timeout)
    except _cf.TimeoutError:
        raise RuntimeError(f"{f.name}: Drive did not respond in {timeout} s. Restart the session, A -> B -> "
                           "this cell (resumes where it stopped).")

def peak(y, dz):
    ok = np.isfinite(y)
    if ok.sum() < 3: return np.nan, True
    k = int(np.nanargmax(np.where(ok, y, -np.inf)))
    if k == 0 or k == len(y) - 1 or not (ok[k-1] and ok[k+1]): return float(dz[k]), True
    a, b, c = y[k-1], y[k], y[k+1]; den = a - 2*b + c
    sub = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
    return float(dz[k] + np.clip(sub, -1, 1)), False

def highpass(v):
    v = v.astype(np.float32)
    return v - ndimage.gaussian_filter(v, (0, HP_SIGMA, HP_SIGMA))

def zshift(a, b, m, zr=None):
    # the k (slices) that makes a[z + k] ~ b[z]. a = our volume, b = victre_fbp. In terms of lesion offsets:
    # k = (lesion offset in a) - (lesion offset in b). zr: the slice range used in b.
    S = a.shape[0]; ha, hb = highpass(a), highpass(b)
    w = m.any(0)
    z0, z1 = zr if zr is not None else (0, S)
    z0, z1 = max(z0, EDGE_Z), min(z1, S - EDGE_Z)
    ks = np.arange(-KMAX, KMAX + 1); c = np.full(len(ks), np.nan)
    for j, k in enumerate(ks):
        zs = [z for z in range(z0, z1) if 0 <= z + k < S]
        if len(zs) < 3: continue
        A = np.stack([ha[z + k][w] for z in zs]); B = np.stack([hb[z][w] for z in zs])
        A = A - A.mean(1, keepdims=True); B = B - B.mean(1, keepdims=True)
        c[j] = float((A * B).sum() / np.sqrt((A * A).sum() * (B * B).sum() + 1e-30))
    k, edge = peak(c, ks)
    return k, edge, float(np.nanmax(c))

def vfbp(d):
    return d["victre_fbp"].astype(np.float32) / 65535.0 * float(d["victre_fbp_scale"])

def geo_d(dn, S, delta):
    # Geometry, with DELTA[dn] changed temporarily. Nothing else changes.
    old = G.DELTA[dn]; G.DELTA[dn] = float(delta)
    try: return G.Geometry(dn, S)
    finally: G.DELTA[dn] = old

def os_sart_ext(geo, proj, valid):
    # the SAME as dataset_generate.Geometry.os_sart; returns the volume uncropped (S + 2m).
    vectors = G.cone_vectors()
    vid = astra.data3d.create("-vol", geo.vol_geom_extended, 0)
    subs = []
    for start in range(G.CFG.sart_subsets):
        views = np.arange(start, G.CFG.num_views, G.CFG.sart_subsets)
        pg = astra.create_proj_geom("cone_vec", G.CFG.det_rows, G.CFG.det_cols, vectors[views])
        subs.append((astra.data3d.create("-proj3d", pg, np.ascontiguousarray(proj[views].transpose(1, 0, 2), np.float32)),
                     astra.data3d.create("-proj3d", pg, np.ascontiguousarray(valid[views].transpose(1, 0, 2), np.float32))))
    for _ in range(G.CFG.sart_iterations):
        for sid, mid in subs:
            cfg = astra.astra_dict("SIRT3D_CUDA")
            cfg["ProjectionDataId"], cfg["ReconstructionDataId"] = sid, vid
            cfg["option"] = {"MinConstraint": 0.0, "SinogramMaskId": mid}
            alg = astra.algorithm.create(cfg); astra.algorithm.run(alg, 1); astra.algorithm.delete(alg)
    v = astra.data3d.get(vid); astra.data3d.delete([vid] + [i for p in subs for i in p])
    return np.ascontiguousarray(v.transpose(0, 2, 1))

def project_ext(geo, vol_ext):
    sid, s = astra.create_sino3d_gpu(np.ascontiguousarray(vol_ext.transpose(0, 2, 1), np.float32),
                                     geo.proj_geom, geo.vol_geom_extended)
    astra.data3d.delete(sid); return s.transpose(1, 0, 2)

def crop(geo, v):
    m = G.CFG.sart_depth_margin_slices; return v[m:m + geo.slices]

def fit(ax, p, sel):
    a, b = ax[sel].astype(np.float64), p[sel].astype(np.float64)
    s = float(a @ b / max(a @ a, 1e-12))
    return s, float(np.linalg.norm(s * a - b) / max(np.linalg.norm(b), 1e-12))

# the same lesion contrast as in 05 (secondary measurement in D)
R_LES = int(round(G.CFG.lesion_radius_mm / (G.CFG.native_pixel_mm * G.CFG.downsample)))
DZ = np.arange(-6, 7)
def contrast(sl, r, c):
    yy, xx = np.ogrid[:sl.shape[0], :sl.shape[1]]
    d2 = (yy - r)**2 + (xx - c)**2
    core, ring = d2 <= R_LES**2, (d2 > (2*R_LES)**2) & (d2 <= (3*R_LES)**2)
    if core.sum() < 5 or ring.sum() < 20: return np.nan
    return float(sl[core].mean() - sl[ring].mean())
def profile(vol, z, r, c):
    zi = int(round(z)); S = vol.shape[0]
    return np.array([contrast(vol[zi + d].astype(np.float64), r, c) if 0 <= zi + d < S else np.nan for d in DZ])
def stacked(profs):
    Y = [y / np.sqrt(np.nanmean(y**2)) for y in profs if np.isfinite(y).sum() >= 5 and np.nanmean(y**2) > 0]
    return peak(np.nanmean(np.stack(Y), 0), DZ)[0] if Y else np.nan

def mean_se(x):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    return (float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan, len(x))
print("ready")
""")

md(r"""
## C · Tissue z offset: `sart_full` ↔ `victre_fbp`, all test phantoms

In each phantom both volumes are high-pass filtered in plane (σ = 4 px), then a slice-by-slice normalised
cross-correlation inside the mask, ±4 slices, parabolic peak. `k > 0`: the tissue in our volume is `k` slices
**deeper** (the same sign as the lesion offset of 05). Also separately by depth third.
One line per phantom in `delta/zshift.jsonl`; finished phantoms are skipped.
""")
code(r"""
PARCA = OUTD/"zshift.jsonl"
done = set()
if PARCA.exists():
    for l in PARCA.read_text().splitlines():
        if l.strip(): done.add(json.loads(l)["patient"])
cand = [f for f in files if f.stem in KT.index and int(KT.loc[f.stem, "victre_fbp"]) == 1]
todo = [f for f in cand if f.stem not in done]
print(f"test phantoms with victre_fbp {len(cand)}, {len(done)} finished, {len(todo)} left", flush=True)
t0 = time.time()
for i, f in enumerate(todo, 1):
    print(f"[{i}/{len(todo)}] {f.stem} ...", end=" ", flush=True)
    d = load_full(f, ["sart_full", "victre_fbp", "victre_fbp_scale", "victre_fbp_present", "breast_mask", "density"])
    if not bool(d["victre_fbp_present"]): print("no victre_fbp"); continue
    a, b, m = d["sart_full"].astype(np.float32), vfbp(d), d["breast_mask"].astype(bool)
    S = a.shape[0]; rec = dict(patient=f.stem, density=str(d["density"]), S=S)
    rec["k"], rec["edge"], rec["c"] = zshift(a, b, m)
    for t in range(3):
        rec[f"k_t{t}"], rec[f"edge_t{t}"], _ = zshift(a, b, m, (S * t // 3, S * (t + 1) // 3))
    with open(PARCA, "a") as h: h.write(json.dumps(rec) + "\n")
    el = time.time() - t0
    print(f"{rec['density']:<9} k={rec['k']:+.2f} (r={rec['c']:.2f}) | ~{el/i*(len(todo)-i)/60:.0f} min left", flush=True)

Z = pd.DataFrame([json.loads(l) for l in PARCA.read_text().splitlines() if l.strip()])
Z.to_csv(OUTD/"zshift.csv", index=False)
CZ = []
for dn in DENS + ["all"]:
    q = Z if dn == "all" else Z[Z.density == dn]
    if len(q) < 2: continue
    row = dict(density=dn, n=len(q), edge=float(q.edge.mean()))
    row["k"], row["k_se"], _ = mean_se(q.k)
    for t in range(3):
        row[f"k_t{t}"], row[f"k_t{t}_se"], _ = mean_se(q[f"k_t{t}"])
    CZ.append(row)
CZ = pd.DataFrame(CZ); CZ.to_csv(OUTD/"zshift_summary.csv", index=False)
pd.set_option("display.width", 200)
print("\n--- tissue offset sart_full - victre_fbp (slices), phantom mean +- s.e. ---")
print(CZ.round(3).to_string(index=False))
print("\nlesion difference of 05 (sart_full - victre_fbp): dense +0.77-1.07 = -0.30 (uncertain), hetero -0.87, scattered -0.64")
""")

md(r"""
## D · DELTA scan

`N_D` test phantoms per density (with victre_fbp and lesions first). For each DELTA, OS-SART is rebuilt
**with the same recipe** (extended volume, cropped), then:

* `k`: tissue offset new volume ↔ `victre_fbp` (the measurement of C). A line `k(DELTA)` is fitted per phantom;
  the DELTA with `k = 0` = **the DELTA that coincides with VICTRE's grid**.
* `k_self`: new volume ↔ the recorded `sart_full` (slices per mm: a scale check).
* stacked lesion offset (against the `.loc`; secondary, few phantoms).
* data-fit residual on the extended volume (for information; not expected to discriminate DELTA, since the margin
  can carry tissue to any height).

Fatty has no `victre_fbp`: only `k_self` and the lesion. Rows per phantom in `delta/scan.jsonl`.
""")
code(r"""
PARCA2 = OUTD/"scan.jsonl"
done2 = set()
if PARCA2.exists():
    for l in PARCA2.read_text().splitlines():
        if l.strip(): done2.add(json.loads(l)["patient"])
SF = []
for dn in DENS:
    c = KT[(KT.density == dn)].sort_values(["victre_fbp", "signal_present"], ascending=False)
    SF += [DATA/"test"/f"{pid}.npz" for pid in c.index[:N_D]]
todo = [f for f in SF if f.stem not in done2]
print(f"{len(SF)} phantoms, {len(todo)} left; DELTA = {D_GRID}", flush=True)
for i, f in enumerate(todo, 1):
    t = time.time()
    d = load_full(f, ["sart_full", "proj_full", "proj_valid", "victre_fbp", "victre_fbp_scale", "victre_fbp_present",
                      "breast_mask", "lesions", "density"])
    dn, gt = str(d["density"]), d["sart_full"].astype(np.float32); S = gt.shape[0]
    p = d["proj_full"].astype(np.float32); valid = d["proj_valid"].astype(bool); sel = valid & (p > 0)
    m = d["breast_mask"].astype(bool); vf = vfbp(d) if bool(d["victre_fbp_present"]) else None
    les = [(s_, r_, c_) for s_, r_, c_, _, kind in d["lesions"] if int(kind) == 1]
    rows = []
    for dl in D_GRID:
        geo = geo_d(dn, S, dl)
        e = os_sart_ext(geo, p, valid); x = np.ascontiguousarray(crop(geo, e))
        r = dict(delta=dl)
        r["k"], r["edge"], _ = zshift(x, vf, m) if vf is not None else (np.nan, True, np.nan)
        r["k_self"], _, _ = zshift(x, gt, m)
        r["lesion"] = stacked([profile(x, *l) for l in les]) if les else np.nan
        r["res_ext"] = fit(project_ext(geo, e), p, sel)[1]
        rows.append(r)
    rec = dict(patient=f.stem, density=dn, S=S, delta_rec=DELTA0[dn], n_mass=len(les), rows=rows,
               lesion_rec=stacked([profile(gt, *l) for l in les]) if les else np.nan)
    with open(PARCA2, "a") as h: h.write(json.dumps(rec, default=float) + "\n")
    print(f"[{i}/{len(todo)}] {f.stem} {dn:<9} {time.time()-t:.0f} s | " +
          "  ".join(f"{r_['delta']}: k={r_['k']:+.2f} les={r_['lesion']:+.2f}" for r_ in rows), flush=True)

SC = []
for l in PARCA2.read_text().splitlines():
    if not l.strip(): continue
    r = json.loads(l)
    for x in r["rows"]:
        SC.append(dict(patient=r["patient"], density=r["density"], delta_rec=r["delta_rec"], lesion_rec=r["lesion_rec"], **x))
SC = pd.DataFrame(SC); SC.to_csv(OUTD/"scan.csv", index=False)

PP = []
for (pid, dn), q in SC.groupby(["patient", "density"]):
    q = q.sort_values("delta"); row = dict(patient=pid, density=dn, delta_rec=float(q.delta_rec.iloc[0]))
    ok = q.k.notna() & ~q.edge.astype(bool)
    if ok.sum() >= 2:
        b, a = np.polyfit(q.delta[ok], q.k[ok], 1)
        row["slope_k"] = b; row["delta_zero"] = -a / b if abs(b) > 0.2 else np.nan
    bs, _ = np.polyfit(q.delta, q.k_self, 1); row["slope_self"] = bs
    PP.append(row)
PP = pd.DataFrame(PP); PP.to_csv(OUTD/"scan_patient.csv", index=False)
print("\n--- per phantom: slope of k(DELTA) (slices/mm) and the DELTA with k = 0 ---")
print(PP.round(3).to_string(index=False))
print("\n--- density mean ---")
DS = []
for dn, q in PP.groupby("density"):
    row = dict(density=dn, delta_rec=float(q.delta_rec.iloc[0]))
    row["delta_zero"], row["delta_zero_se"], row["n"] = mean_se(q.get("delta_zero", pd.Series(dtype=float)))
    row["slope_self"], _, _ = mean_se(q.slope_self)
    les = SC[SC.density == dn].groupby("delta").lesion.mean()
    for dl, v in les.items(): row[f"lesion@{dl}"] = v
    row["lesion@record"] = float(SC[SC.density == dn].groupby("patient").lesion_rec.first().mean())
    res = SC[SC.density == dn].groupby("delta").res_ext.median()
    for dl, v in res.items(): row[f"res@{dl}"] = v
    DS.append(row)
DS = pd.DataFrame(DS); DS.to_csv(OUTD/"scan_summary.csv", index=False)
print(DS.round(3).T.to_string())
""")

md(r"""
## E · Mask edge

Where are the low `sart_full` voxels of 05 F (`< 0.02 mm⁻¹`, inside the mask)? Distance to the in-plane edge of
the mask (distance transform per slice) and depth. Comparison: all mask voxels.
`breast_mask = hull ∧ (5×5-smoothed sart_full > 0.08·p99.5)`; does the smoothing leave a band at the edge?
""")
code(r"""
ME = []
for i, f in enumerate(SF, 1):
    d = load_full(f, ["sart_full", "breast_mask", "density"])
    gt, m, dn = d["sart_full"].astype(np.float32), d["breast_mask"].astype(bool), str(d["density"])
    S = gt.shape[0]
    dist = np.stack([ndimage.distance_transform_edt(m[z]) for z in range(S)])
    zf = ((np.arange(S) + 0.5) / S)[:, None, None] * np.ones_like(gt)
    low = m & (gt < 0.02)
    edge_z = (zf < 0.1) | (zf > 0.9)
    for tag, sel in (("low", low), ("all", m)):
        n = int(sel.sum())
        ME.append(dict(patient=f.stem, density=dn, set=tag, n=n,
                       frac_d1=float((dist[sel] <= 1).mean()) if n else np.nan,
                       frac_d2=float((dist[sel] <= 2).mean()) if n else np.nan,
                       frac_d5=float((dist[sel] <= 5).mean()) if n else np.nan,
                       frac_zedge=float(edge_z[sel].mean()) if n else np.nan,
                       frac_inner=float((~edge_z[sel] & (dist[sel] > 5)).mean()) if n else np.nan))
    print(f"[{i}/{len(SF)}] {f.stem}", flush=True)
ME = pd.DataFrame(ME); ME.to_csv(OUTD/"mask_edge.csv", index=False)
print("\n--- low voxels (low) vs whole mask (all): <=1/2/5 px from the edge, at the depth ends (10 %), neither ---")
print(ME.groupby(["density", "set"])[["frac_d1", "frac_d2", "frac_d5", "frac_zedge", "frac_inner"]].median().round(3).to_string())
""")

md("## F · Decision → `delta/summary.json`")
code(r"""
CZ = pd.read_csv(OUTD/"zshift_summary.csv"); DS = pd.read_csv(OUTD/"scan_summary.csv")
ME = pd.read_csv(OUTD/"mask_edge.csv")
oz = {"delta_rec": DELTA0, "tissue_shift": {}, "scan": {}, "verdict": {}}
slope = float(DS.slope_self.mean())            # slices / mm (expected ~ -1)
for r in CZ.itertuples():
    oz["tissue_shift"][r.density] = dict(k=r.k, se=r.k_se, n=r.n, terciles=[getattr(r, f"k_t{t}") for t in range(3)])
for r in DS.itertuples():
    oz["scan"][r.density] = dict(delta_zero=r.delta_zero, se=r.delta_zero_se, n=r.n, slope_self=r.slope_self)
print(f"scale check: when DELTA grows by 1 mm the volume shifts by {slope:+.2f} slices (expected ~ -1)\n")
for dn in DENS:
    rec = DELTA0[dn]
    cz = CZ[CZ.density == dn]; ds = DS[DS.density == dn]
    lines = [f"{dn}: DELTA in the record {rec}"]
    if len(cz):
        k, se = float(cz.k.iloc[0]), float(cz.k_se.iloc[0])
        pred = k + slope * (19.5 - rec)
        lines.append(f"  C  tissue offset (sart_full - victre_fbp) {k:+.2f} +- {se:.2f} slices; expected at DELTA=19.5 {pred:+.2f}")
        oz["verdict"][dn] = dict(k=k, k_se=se, k_pred_195=pred)
    if len(ds) and np.isfinite(ds.delta_zero.iloc[0]):
        z0, se0 = float(ds.delta_zero.iloc[0]), float(ds.delta_zero_se.iloc[0])
        tol = max(2 * se0, 0.15) if np.isfinite(se0) else 0.15
        near = "19.5 (= the record)" if rec == 19.5 else ("19.5" if abs(z0 - 19.5) < abs(z0 - rec) else "the recorded")
        ok195 = abs(z0 - 19.5) <= tol
        okrec = abs(z0 - rec) <= tol
        lines.append(f"  D  DELTA coinciding with VICTRE {z0:.2f} +- {se0:.2f} mm -> closer to {near} value "
                     f"(consistent with 19.5: {'YES' if ok195 else 'NO'}, consistent with the record: {'YES' if okrec else 'NO'})")
        oz["verdict"].setdefault(dn, {}).update(delta_zero=z0, se=se0, ok_195=bool(ok195), ok_rec=bool(okrec))
    print("\n".join(lines))
lo = ME[ME.set == "low"].groupby("density")[["frac_d2", "frac_zedge", "frac_inner"]].median()
al = ME[ME.set == "all"].groupby("density")[["frac_d2", "frac_zedge"]].median()
oz["mask_edge"] = {"low": lo.round(4).to_dict(), "all": al.round(4).to_dict()}
print("\nmask: fraction of the low voxels in the edge band (<=2 px)", lo.frac_d2.round(2).to_dict(),
      "| in the whole mask", al.frac_d2.round(2).to_dict())
(OUTD/"summary.json").write_text(json.dumps(oz, indent=1, default=float))
print("\nwritten:", OUTD/"summary.json")
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
