"""Generates 03_extra_analysis_colab.ipynb. Analyses for the figure plan of the Data Descriptor that 01 does not make."""
import json, sys
from pathlib import Path

OUT = Path(sys.argv[1])
cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md(r"""
# VICTRE-Paired v11 — extra analysis (Colab, GPU)

Measurements for the figure plan of the Data Descriptor that **01 does not make**. Independent of 01,
it can run at the same time; 02 merges the outputs of both.

| | cell | what it does | time |
|---|---|---|---|
| **A** | setup | GPU, ASTRA, Drive | 2 min |
| **B** | settings | paths, methods, helpers | 1 min |
| **C** | checks + settings | correctness of the methods; the SART pass count and the SART-TV weight are chosen on **validation** | ~15 min |
| **D** | test analysis | per phantom, 4 methods × 3 doses × 2 projection sources, and the measures | ~2 min/phantom |
| **E** | collect + check | CSV files, cross-check with 01 | 2 min |

**Methods** (all with the same ASTRA geometry, from `dataset_generate.py`):

| name | what | note |
|---|---|---|
| `FBP` | `Geometry.fbp` | as in 01 |
| `SART` | one view per subset (25 subsets) | pass count chosen on validation |
| `OS-SART` | 5 subsets × 10 passes | the same recipe that produces `sart_full` |
| `SART-TV` | OS-SART + an in-plane TV descent after each pass (ASD-POCS type) | TV weight chosen on validation |

**Projection sources:** `mc` = VICTRE's Monte Carlo projections (`proj_*` in the dataset);
`syn` = `A·sart_full`, noised with the same dose model (the classical inverse-crime regime).

**Scope:** the ranking under the two sources is **reported**; its cause is the subject of the companion paper.
Since `sart_full` is an OS-SART reconstruction, the OS-SART family has an advantage against it;
the Data Descriptor states this in one sentence and gives the second reference (`victre_fbp`) alongside.

**D resumes where it stopped** (`extra/phantoms/<id>.json`). If the session drops: **A → B → D**.
Two tabs: `SESSION = 0/1`, `N_SESSIONS = 2`. **First a trial:** C → D → E with `TRIAL = 3`.

Outputs: `/content/drive/MyDrive/victre_paired/analysis/extra/`
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

md(r"""
## B · SETTINGS
""")
code(r"""
import os, sys, json, time, collections
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage

NB_DIR = Path("/content/drive/MyDrive/victre_paired/code")
DATA   = Path("/content/drive/MyDrive/victre_paired/data")
RES    = Path("/content/drive/MyDrive/victre_paired/analysis")
EK     = RES/"extra"

TRIAL     = 0        # >0: only this many test phantoms (trial). 0: all
SESSION    = 0
N_SESSIONS = 1

for d in (EK, EK/"phantoms", EK/"panel"):
    d.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(NB_DIR)); os.chdir(NB_DIR)
import dataset_generate as G
import dataset_test as T
assert all(v == 19.5 for v in G.DELTA.values()), f"dataset_generate.py on Drive is old (before the DELTA patch): {G.DELTA}"
%matplotlib inline

DENS, DOSES = G.DENSITIES, T.DOSES                      # ['50','25','10']
METHODS = ["FBP", "SART", "OS-SART", "SART-TV"]
SOURCES = ["mc", "syn"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
PIX = G.CFG.native_pixel_mm * G.CFG.downsample           # 0.34 mm
files = {s: sorted(p for p in (DATA/s).glob("*.npz") if ".partial" not in p.name) for s in ("val", "test")}
print({s: len(v) for s, v in files.items()})

# ------------------------------------------------------------------ reconstruction
def tv_grad2d(x, eps=1e-10):
    # Gradient of in-plane isotropic TV (each slice separately): depth resolution is poor in DBT,
    # so TV acts only within the slice. Forward differences, Neumann boundary.
    dx = torch.zeros_like(x); dx[:, :, :-1] = x[:, :, 1:] - x[:, :, :-1]
    dy = torch.zeros_like(x); dy[:, :-1, :] = x[:, 1:, :] - x[:, :-1, :]
    n = torch.sqrt(dx * dx + dy * dy + eps)
    px, py = dx / n, dy / n
    div = px.clone(); div[:, :, 1:] -= px[:, :, :-1]
    div += py; div[:, 1:, :] -= py[:, :-1, :]
    return -div

def tv_steps(x, x_prev, alpha, n_tv=10):
    # ASD-POCS (Sidky & Pan 2008) type: the TV step size is proportional to the size of the SART step
    # in the same pass (alpha is dimensionless). Non-negativity is kept.
    t = torch.from_numpy(np.ascontiguousarray(x)).to(DEV)
    dp = float(torch.linalg.norm(t - torch.from_numpy(np.ascontiguousarray(x_prev)).to(DEV)))
    step = alpha * dp
    for _ in range(n_tv):
        g = tv_grad2d(t); ng = float(torch.linalg.norm(g))
        if ng > 0: t = t - step * g / ng
    return torch.clamp(t, min=0).cpu().numpy().astype(np.float32)

def iterative(geo, proj, valid, subsets, iterations, tv_alpha=0.0, n_tv=10):
    # the same as G.Geometry.os_sart, with a choice of subset count and TV. subsets=5, iterations=10,
    # tv_alpha=0 -> the sart_full recipe (checked exactly in C).
    vectors = G.cone_vectors(); m = G.CFG.sart_depth_margin_slices
    vid = astra.data3d.create("-vol", geo.vol_geom_extended, 0)
    subs = []
    for start in range(subsets):
        views = np.arange(start, G.CFG.num_views, subsets)
        geom = astra.create_proj_geom("cone_vec", G.CFG.det_rows, G.CFG.det_cols, vectors[views])
        data = np.ascontiguousarray(proj[views].transpose(1, 0, 2), np.float32)
        mask = np.ascontiguousarray(valid[views].transpose(1, 0, 2), np.float32)
        subs.append((astra.data3d.create("-proj3d", geom, data), astra.data3d.create("-proj3d", geom, mask)))
    try:
        for _ in range(iterations):
            x0 = astra.data3d.get(vid) if tv_alpha else None
            for sid, mid in subs:
                cfg = astra.astra_dict("SIRT3D_CUDA")
                cfg["ProjectionDataId"], cfg["ReconstructionDataId"] = sid, vid
                cfg["option"] = {"MinConstraint": 0.0, "SinogramMaskId": mid}
                aid = astra.algorithm.create(cfg); astra.algorithm.run(aid, 1); astra.algorithm.delete(aid)
            if tv_alpha:
                astra.data3d.store(vid, tv_steps(astra.data3d.get(vid), x0, tv_alpha, n_tv))
        vol = astra.data3d.get(vid)
    finally:
        astra.data3d.delete([vid] + [i for p in subs for i in p])
    return np.ascontiguousarray(vol[m:m + geo.slices].transpose(0, 2, 1))

def reconstruct(geo, method, proj, valid, settings):
    if method == "FBP":     return geo.fbp(proj)
    if method == "OS-SART": return iterative(geo, proj, valid, G.CFG.sart_subsets, G.CFG.sart_iterations)
    if method == "SART":    return iterative(geo, proj, valid, G.CFG.num_views, settings["sart_iterations"])
    if method == "SART-TV": return iterative(geo, proj, valid, G.CFG.sart_subsets, G.CFG.sart_iterations,
                                             tv_alpha=settings["tv_alpha"])
    raise ValueError(method)

# ------------------------------------------------------------------ synthetic projections
def synthetic_full(geo, gt, footprint, valid):
    # A x_ref, with the same footprint and valid rays as the MC projections.
    return np.where(footprint & valid, np.clip(geo.project(gt), 0, None), 0.0).astype(np.float32)

def synth_dose(p, footprint, n0, sigma_e, f, rng):
    # Measurement at dose f directly from the noiseless p: N = f N0 e^-p, Var = N + sigma_e^2 (the same
    # distribution the dataset's dose model targets; on the MC side the full dose carries its own noise).
    counts = f * n0 * np.exp(-p.astype(np.float64))
    meas = counts + rng.standard_normal(p.shape) * np.sqrt(counts + sigma_e**2)
    return np.where(footprint, np.log(f * n0 / np.clip(meas, G.CFG.min_counts, None)), 0.0).astype(np.float32)

# ------------------------------------------------------------------ measurement helpers (as in 01)
R_CORE, R_BG = 4, 12
def sdnr_at(vol, z, h, w):
    z, h, w = int(round(z)), int(round(h)), int(round(w))
    if not (0 <= z < vol.shape[0] and 0 <= h < vol.shape[1] and 0 <= w < vol.shape[2]):
        return None
    core = vol[max(0, z-1):z+2, max(0, h-R_CORE):h+R_CORE+1, max(0, w-R_CORE):w+R_CORE+1]
    bg   = vol[z, max(0, h-R_BG):h+R_BG+1, max(0, w-R_BG):w+R_BG+1]
    s = float(bg.std())
    return None if s < 1e-12 else float((core.mean() - bg.mean()) / s)

def corr_masked(a, b, m):
    x = a[m].astype(np.float64); y = b[m].astype(np.float64)
    x -= x.mean(); y -= y.mean()
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))

COUNT_EDGES = np.logspace(2, 5, 16)          # bins of full-dose photon counts (Fig 5a)
def noise_bins(full, low, n0, sigma_e, f):
    # the same measurement as dataset_test.noise_ratios, binned by count: measured extra log variance
    # and the model's prediction (f(1-f)N + (1-f^2) se^2) / (fN)^2.
    inner = ndimage.binary_erosion(full > 0, iterations=6, structure=np.ones((1, 3, 3)))
    counts = n0 * np.exp(-full[inner].astype(np.float64))
    pred = (f * (1 - f) * counts + (1 - f**2) * sigma_e**2) / (f * counts) ** 2
    meas = (G.local_variance(low) - G.local_variance(full))[inner]
    k = np.digitize(counts, COUNT_EDGES) - 1
    out = []
    for b in range(len(COUNT_EDGES) - 1):
        s = k == b
        if s.sum() >= 200:
            out.append(dict(bin=b, n=int(s.sum()), counts=float(np.median(counts[s])),
                            measured=float(np.median(meas[s])), predicted=float(np.median(pred[s]))))
    return out

NPS_N = 64
def nps_radial(noise, mask):
    # In-plane noise power spectrum: 64x64 patches inside the mask, mean removed,
    # |FFT|^2 * pixel area / N^2, radial average. Unit: mm^-2 * mm^2 (value^2 mm^2).
    inner = ndimage.binary_erosion(mask, iterations=int(round(5 / PIX)))
    P, acc = NPS_N, []
    for z in range(noise.shape[0]):
        for r in range(0, noise.shape[1] - P, P // 2):
            for c in range(0, noise.shape[2] - P, P // 2):
                if inner[z, r:r+P, c:c+P].all():
                    patch = noise[z, r:r+P, c:c+P].astype(np.float64)
                    acc.append(np.abs(np.fft.fft2(patch - patch.mean())) ** 2 * PIX**2 / P**2)
    if not acc: return None, 0
    nps = np.mean(acc, 0)
    fy, fx = np.meshgrid(np.fft.fftfreq(P), np.fft.fftfreq(P), indexing="ij")
    fr = np.sqrt(fx**2 + fy**2); kb = np.minimum((fr * P).astype(int), P // 2)
    return [float(nps[kb == i].mean()) for i in range(P // 2)], len(acc)

NPS_FREQ = [i / NPS_N / PIX for i in range(NPS_N // 2)]   # cycles/mm
print("ready. methods", METHODS, "| sources", SOURCES, "| device", DEV)
""")

md(r"""
## C · Checks and settings — once, before D

Four checks, all must be **YES**; otherwise D does not run:

1. `iterative(5 subsets, 10 passes)` equals the dataset's recipe (`Geometry.os_sart`)
2. The noise added to the synthetic projections matches the dose model (ratio ≈ 1)
3. The TV step really has an effect (SART-TV ≠ OS-SART)
4. The chosen settings are not at the edge of the grid (if they are, the grid is extended in that direction automatically)

**The settings are chosen on the validation split** (one signal-present phantom per density, 25 % dose, MC source):
the SART pass count and the SART-TV weight that give the highest PSNR against `sart_full`. The test
split is not looked at. The choice is written to `extra/settings.json`, and D reads it from there.
""")
code(r"""
ok, t0 = {}, time.time()
def pick_val():
    out = {}
    for f in files["val"]:
        with np.load(f) as z:
            dn = str(z["density"])
            if dn in out or not bool(z["signal_present"]): continue
            out[dn] = f
        if len(out) == len(DENS): break
    return [out[dn] for dn in DENS if dn in out]
VAL = pick_val()
print("validation phantoms:", [f.stem for f in VAL])

def load(f, tries=5):
    # The Drive (FUSE) connection can drop in long runs -> "Software caused connection abort",
    # followed by BadZipFile. The file is not corrupt: wait, remount Drive, try again.
    for t in range(tries):
        try:
            with np.load(f) as z:
                return {k: z[k] for k in z.files}
        except Exception as e:
            if t == tries - 1: raise
            print(f"   read error ({type(e).__name__}), retrying in {30*(t+1)} s: {f.name}", flush=True)
            time.sleep(30 * (t + 1))
            try:
                from google.colab import drive as _d; _d.mount("/content/drive", force_remount=True)
            except Exception:
                pass

# 1. recipe
d = load(VAL[0]); geo = G.Geometry(str(d["density"]), d["sart_full"].shape[0])
p25 = d["proj_25"].astype(np.float32)
a, b = iterative(geo, p25, d["proj_valid"], 5, 10), geo.os_sart(p25, d["proj_valid"])
c1 = corr_masked(a, b, d["breast_mask"])
print(f"1. iterative(5,10) vs Geometry.os_sart: corr = {c1:.7f}  max difference {np.abs(a-b).max():.2e}")
ok["1 recipe"] = c1 > 0.99999

# 2. synthetic noise
fp = d["proj_full"].astype(np.float32) > 0
ps = synthetic_full(geo, d["sart_full"], fp, d["proj_valid"])
n0, se = float(d["photons_per_pixel"]), float(d["electronic_sigma"])
low = synth_dose(ps, fp, n0, se, 0.25, np.random.default_rng(0))
inner = ndimage.binary_erosion(fp, iterations=6, structure=np.ones((1, 3, 3)))
cnt = 0.25 * n0 * np.exp(-ps[inner].astype(np.float64))
pred = (cnt + se**2) / cnt**2
meas = (G.local_variance(low) - G.local_variance(ps))[inner]
r2 = float(np.median(meas) / np.median(pred))
print(f"2. synthetic noise ratio (measured/predicted) = {r2:.3f}   (the estimator of 01 gives ~0.93-0.97)")
ok["2 synthetic"] = 0.85 < r2 < 1.15

# choice of settings
VD = [load(f) for f in VAL]
def val_psnr(fn):
    ps_ = []
    for dd in VD:
        g_ = G.Geometry(str(dd["density"]), dd["sart_full"].shape[0])
        ps_.append(T.metrics(fn(g_, dd["proj_25"].astype(np.float32), dd["proj_valid"]), dd["sart_full"], dd["breast_mask"])["psnr"])
    return float(np.mean(ps_))
# If the grid edge wins, the grid is extended in that direction (until the best value is an interior point).
def search(fn, grid, step, lo_lim, hi_lim, cache):
    grid = sorted(grid)
    while True:
        for v in grid:
            if v not in cache: cache[v] = fn(v)
        best = max(grid, key=lambda v: cache[v])
        if best == grid[-1] and step(best, +1) <= hi_lim: grid.append(step(best, +1)); continue
        if best == grid[0] and step(best, -1) >= lo_lim and step(best, -1) not in grid: grid.insert(0, step(best, -1)); continue
        return best, [(v, cache[v]) for v in grid]

sart_cache = {}
sart_n, sart_tab = search(lambda n: val_psnr(lambda g_, p_, v_: iterative(g_, p_, v_, G.CFG.num_views, n)),
                          (1, 2, 3, 4), lambda v, d: v + d, 1, 12, sart_cache)
print("\nSART pass count (validation, 25 %, PSNR vs sart_full):")
for n, v in sart_tab: print(f"   {n} passes  {v:6.2f} dB" + ("   <-- chosen" if n == sart_n else ""))
tv_cache = {}
tv_a, tv_tab = search(lambda al: val_psnr(lambda g_, p_, v_: iterative(g_, p_, v_, 5, 10, tv_alpha=al)),
                      (0.02, 0.05, 0.1, 0.2, 0.4, 0.8), lambda v, d: round(v * 2.0 ** d, 6), 0.0025, 12.8, tv_cache)
os_ref = val_psnr(lambda g_, p_, v_: iterative(g_, p_, v_, 5, 10))
print(f"\nSART-TV weight (validation, 25 %; OS-SART = {os_ref:.2f} dB):")
for al, v in tv_tab: print(f"   alpha {al:<8g} {v:6.2f} dB" + ("   <-- chosen" if al == tv_a else ""))
SART_GRID, TV_GRID = [n for n, _ in sart_tab], [a for a, _ in tv_tab]

x_tv = iterative(geo, p25, d["proj_valid"], 5, 10, tv_alpha=tv_a)
d3 = float(np.abs(x_tv - a)[d["breast_mask"]].mean() / np.abs(a)[d["breast_mask"]].mean())
print(f"\n3. difference between SART-TV and OS-SART = {d3:.4f}  (should be > 0.001)")
ok["3 TV effective"] = d3 > 1e-3
# edge: if the extension limit was also exceeded (SART 12 passes, alpha 0.0025 / 12.8), it really is the edge
ok["4 not at edge"] = sart_n != SART_GRID[-1] and tv_a not in (TV_GRID[0], TV_GRID[-1])

print()
for k, v in ok.items(): print(f"{k:<14} -> {'YES' if v else 'NO  <-- CHECK'}")
settings = dict(sart_iterations=int(sart_n), tv_alpha=float(tv_a), sart_table=sart_tab, tv_table=tv_tab,
            os_sart_val_psnr=os_ref, recipe_corr=c1, synthetic_noise_ratio=r2, tv_effect=d3,
            val_patients=[f.stem for f in VAL], ok={k: bool(v) for k, v in ok.items()})
_ = (EK/"settings.json").write_text(json.dumps(settings, indent=1))
print(f"\n{(time.time()-t0)/60:.1f} min. If 4 is NO, the grid limit was exceeded: report the output.")
""")

md(r"""
## D · Test analysis — per phantom

For each test phantom:

- **reconstruction:** 4 methods × 3 doses × 2 sources (`mc`, `syn`) = 24 volumes, plus full-dose FBP (for the noise spectrum)
- **agreement:** PSNR, SSIM (`dataset_test.metrics`) and correlation against `sart_full`; correlation against `victre_fbp` where present (**reported only**)
- **detection:** SDNR at every lesion and control location (the same definition and same-location rule as in 01)
- **forward model:** per view `‖p_mc − s·A·sart_full‖ / ‖p_mc‖` (Fig 2c-d)
- **noise:** extra variance binned by count (Fig 5a); noise power spectrum for OS-SART and FBP (Fig 5b)
- **panel:** slices for one signal-present phantom per density (Fig 3, 6, 7; those with `victre_fbp` preferred)
""")
code(r"""
settings = json.loads((EK/"settings.json").read_text())
assert all(settings["ok"].values()), f"the C checks did not pass: {settings['ok']}"
print("settings:", {k: settings[k] for k in ("sart_iterations", "tv_alpha")})

work = [f for i, f in enumerate(files["test"]) if i % N_SESSIONS == SESSION]
if TRIAL: work = work[:TRIAL]
todo = [f for f in work if not (EK/"phantoms"/f"{f.stem}.json").exists()]
print(f"test: {len(work)} phantoms in this tab, {len(work)-len(todo)} finished, {len(todo)} left\n", flush=True)

def want_panel(dn, sp, has_vf, les):
    if not sp or not (les[:, 4] == 1).any() or not (les[:, 4] == 0).any(): return False
    p = EK/"panel"/f"{dn}.npz"
    if not p.exists(): return True
    with np.load(p) as old:
        return has_vf and not bool(old["victre_fbp_present"])

def crop2(img, r, c, half):
    r, c = int(round(r)), int(round(c))
    out = np.zeros((2*half, 2*half), np.float32)
    r0, c0 = r - half, c - half
    rs, cs = slice(max(r0, 0), min(r0 + 2*half, img.shape[0])), slice(max(c0, 0), min(c0 + 2*half, img.shape[1]))
    out[rs.start - r0:rs.stop - r0, cs.start - c0:cs.stop - c0] = img[rs, cs]
    return out

def core_profile(vol, r, c):
    r, c = int(round(r)), int(round(c))
    return vol[:, max(0, r-R_CORE):r+R_CORE+1, max(0, c-R_CORE):c+R_CORE+1].mean((1, 2)).astype(np.float32)

t_all = time.time()
for i, f in enumerate(todo):
    t = time.time()
    d = load(f)
    dn, gt, m = str(d["density"]), d["sart_full"], d["breast_mask"]
    S, les, valid = gt.shape[0], d["lesions"], d["proj_valid"]
    geo = G.Geometry(dn, S)
    pf = d["proj_full"].astype(np.float32); fp = pf > 0
    n0, se = float(d["photons_per_pixel"]), float(d["electronic_sigma"])
    has_vf = bool(d["victre_fbp_present"])
    vf = (d["victre_fbp"].astype(np.float32) / 65535.0 * float(d["victre_fbp_scale"])) if has_vf else None

    # --- projections
    ps_full = synthetic_full(geo, gt, fp, valid)
    rng = np.random.default_rng(10_000 + int(f.stem.lstrip("-")) % 100_000)
    PROJ = {("mc", k): d[f"proj_{k}"].astype(np.float32) for k in DOSES}
    PROJ.update({("syn", k): synth_dose(ps_full, fp, n0, se, float(k) / 100, rng) for k in DOSES})

    # --- forward-model residual, per view
    vv = valid & fp
    s_ = float((ps_full[vv].astype(np.float64) @ pf[vv]) / max(ps_full[vv].astype(np.float64) @ ps_full[vv], 1e-12))
    view_res = [float(np.linalg.norm((pf[v] - s_ * ps_full[v])[vv[v]]) / max(np.linalg.norm(pf[v][vv[v]]), 1e-12))
                for v in range(pf.shape[0])]

    # --- reconstructions
    vols = {(src, meth, k): reconstruct(geo, meth, PROJ[(src, k)], valid, settings)
            for src in SOURCES for meth in METHODS for k in DOSES}
    fbp_full = geo.fbp(pf)

    rec = dict(patient=f.stem, split="test", density=dn, signal_present=int(bool(d["signal_present"])),
               victre_fbp_present=int(has_vf), slices=S, metrics=[], sdnr=[], view_res=view_res, view_scale=s_)
    for (src, meth, k), v in vols.items():
        mm = T.metrics(v, gt, m)
        rec["metrics"].append(dict(source=src, method=meth, dose=k, psnr=float(mm["psnr"]), ssim=float(mm["ssim"]),
                                   corr=corr_masked(v, gt, m), corr_vf=(corr_masked(v, vf, m) if has_vf else None)))

    # --- detection (the rule of 01: a location counts only if it is defined in the GT and in every volume)
    allv = {("GT", "GT", "100"): gt, **vols}
    for s0, r0, c0, num, kind in les:
        xs = {kk: sdnr_at(v, s0, r0, c0) for kk, v in allv.items()}
        if any(x is None for x in xs.values()): continue
        for (src, meth, k), x in xs.items():
            rec["sdnr"].append(dict(source=src, method=meth, dose=k, kind=int(kind), num=int(num), sdnr=x))

    # --- noise: bins (MC projections) and NPS (MC; OS-SART and FBP, relative to the same method at full dose)
    rec["noise_bins"] = {k: noise_bins(pf, PROJ[("mc", k)], n0, se, float(k) / 100) for k in DOSES}
    rec["nps"] = {}
    for meth, full in (("OS-SART", gt), ("FBP", fbp_full)):
        for k in DOSES:
            v = vols[("mc", meth, k)]
            if meth == "FBP":     # FBP is not in mm^-1: first scale-matched to its full dose
                v = T.scale_match(v, full, m)
            spec, npatch = nps_radial(v - full, m)
            if spec: rec["nps"][f"{meth}|{k}"] = dict(nps=spec, patches=npatch)

    # --- panel
    sp = bool(d["signal_present"])
    if want_panel(dn, sp, has_vf, les):
        mass = les[les[:, 4] == 1][0]; calc = les[les[:, 4] == 0][0]
        zm, rm, cm = int(round(mass[0])), mass[1], mass[2]
        zc, rc, cc = int(round(calc[0])), calc[1], calc[2]
        HM, HC = int(round(20 / PIX)), int(round(10 / PIX))           # 40 mm and 20 mm squares
        P = dict(patient=f.stem, density=dn, slices=S, z=zm, mass=mass, calc=calc, lesions=les,
                 mask=m[zm], mask_calc=m[zc], sart_full=gt[zm], depth_cut=gt[:, int(round(rm)), :],
                 depth_mask=m[:, int(round(rm)), :], lesion_roi=d["lesion_roi"][zm],
                 victre_fbp_present=has_vf, victre_fbp=(vf[zm] if has_vf else np.zeros((0, 0), np.float32)),
                 gt_mass=crop2(gt[zm], rm, cm, HM), gt_calc=crop2(gt[zc], rc, cc, HC),
                 prof_gt=core_profile(gt, rm, cm),
                 prof_vf=(core_profile(vf, rm, cm) if has_vf else np.zeros(0, np.float32)),
                 proj_full=d["proj_full"][12], proj_valid=d["proj_valid"][12],
                 view00_mc=d["proj_full"][0], view24_mc=d["proj_full"][24], view00_syn=(s_ * ps_full[0]).astype(np.float16),
                 view12_syn=(s_ * ps_full[12]).astype(np.float16), valid00=d["proj_valid"][0],
                 dm_full=d["dm_full"], dm_valid=d["dm_valid"],
                 **{f"proj_{k}": d[f"proj_{k}"][12] for k in DOSES}, **{f"dm_{k}": d[f"dm_{k}"] for k in DOSES})
        for meth in METHODS:
            for k in DOSES:
                v = vols[("mc", meth, k)]
                key = f"{meth.replace('-', '')}_{k}"
                if meth == "FBP": v = T.scale_match(v, gt, m)
                P[f"mass_{key}"] = crop2(v[zm], rm, cm, HM)
                P[f"calc_{key}"] = crop2(v[zc], rc, cc, HC)
                P[f"prof_{key}"] = core_profile(v, rm, cm)
                P[f"slice_{key}"] = v[zm].astype(np.float16)
        np.savez_compressed(EK/"panel"/f"{dn}.npz", **P)

    rec["seconds"] = round(time.time() - t, 1)
    tmp = EK/"phantoms"/f"{f.stem}.json.tmp"
    tmp.write_text(json.dumps(rec, default=float)); tmp.rename(EK/"phantoms"/f"{f.stem}.json")
    el = time.time() - t_all
    print(f"[{i+1}/{len(todo)}] {f.stem} {dn:<9} vfbp={int(has_vf)}  {rec['seconds']:.0f} s"
          f"  | ~{el/(i+1)*(len(todo)-i-1)/3600:.1f} h left", flush=True)
print(f"\ndone: {len(todo)} phantoms, {(time.time()-t_all)/3600:.2f} h")
""")

md(r"""
## E · Collect and check

`extra/phantoms/*.json` → `extra_metrics.csv`, `extra_sdnr.csv`, `extra_viewres.csv`, `extra_noisebins.csv`, `extra_nps.csv`.
Cross-check: the PSNR of OS-SART on the MC source must equal that of SART in the `baseline.csv` of 01
(same recipe, same projections).
""")
code(r"""
import pandas as pd
recs = [json.loads(p.read_text()) for p in sorted((EK/"phantoms").glob("*.json"))]
assert recs, "no phantoms -> run D first"
BASE = ["patient", "density", "signal_present", "victre_fbp_present"]
M  = pd.DataFrame([{**{k: r[k] for k in BASE}, **x} for r in recs for x in r["metrics"]])
SD = pd.DataFrame([{**{k: r[k] for k in BASE}, **x} for r in recs for x in r["sdnr"]])
VR = pd.DataFrame([{**{k: r[k] for k in BASE}, "view": v, "res": x, "scale": r["view_scale"]}
                   for r in recs for v, x in enumerate(r["view_res"])])
NB = pd.DataFrame([{**{k: r[k] for k in BASE}, "dose": k, **b} for r in recs for k, bs in r["noise_bins"].items() for b in bs])
NP = pd.DataFrame([{**{k: r[k] for k in BASE}, "method": key.split("|")[0], "dose": key.split("|")[1],
                    "freq": NPS_FREQ[j], "nps": v, "patches": x["patches"]}
                   for r in recs for key, x in r["nps"].items() for j, v in enumerate(x["nps"])])
for name, df in (("extra_metrics", M), ("extra_sdnr", SD), ("extra_viewres", VR), ("extra_noisebins", NB), ("extra_nps", NP)):
    df.to_csv(RES/f"{name}.csv", index=False); print(f"{name+'.csv':<18} {len(df):>8} rows")
print(f"\n{len(recs)} phantoms, mean {np.mean([r['seconds'] for r in recs]):.0f} s/phantom")

def ok(c): return "OK" if c else "CHECK  <--"
print("\n--- PSNR vs sart_full (mean), source x method x dose ---")
print(M.pivot_table(index=["source", "method"], columns="dose", values="psnr", aggfunc="mean")[DOSES].round(2))

base = RES/"baseline.csv"
if base.exists():
    B = pd.read_csv(base); B["dose"] = B["dose"].astype(str)
    common = set(M.patient.astype(str)) & set(B.patient.astype(str))
    b = B[(B.method == "SART") & (B.ref == "sart_full") & B.patient.astype(str).isin(common)].groupby("dose").psnr.mean()
    e = M[(M.source == "mc") & (M.method == "OS-SART") & M.patient.astype(str).isin(common)].groupby("dose").psnr.mean()
    for k in DOSES:
        print(f"cross-check {k}%: 03 OS-SART {e[k]:.3f} | 01 SART {b[k]:.3f}  -> {ok(abs(e[k]-b[k]) < 0.05)}")
else:
    print("no baseline.csv (01-E has not run): cross-check skipped")

print("\n--- ranking (PSNR, 25 %): by source ---")
for src in SOURCES:
    g = M[(M.source == src) & (M.dose == "25")].groupby("method").psnr.mean().sort_values(ascending=False)
    print(f"{src}: " + "  >  ".join(f"{k} {v:.2f}" for k, v in g.items()))
print("\n--- forward-model residual (MC vs A*sart_full), median ---")
print(VR.groupby("density").res.median().round(4).to_dict(), "| scale s median", round(float(VR.scale.median()), 4))
print("\n--- control SDNR (should be ~0), MC 25 % ---")
g = SD[(SD.source == "mc") & (SD.dose == "25") & (SD.kind == -1)].groupby("method").sdnr.mean()
print(g.round(3).to_dict())
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
