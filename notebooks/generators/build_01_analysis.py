"""Generates 01_analysis_colab.ipynb. The cells are below as plain text; line endings follow the notebook spec."""
import json, sys
from pathlib import Path

OUT = Path(sys.argv[1])
cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md(r"""
# VICTRE-Paired v11 — analysis for the Data Descriptor (Colab, GPU)

**When:** after generation (E) and validation (F → F2 → G) have finished.
This notebook produces **all the numbers** of the Data Descriptor; its figures and tables are drawn from them by
`02_figures_tables_colab.ipynb`.

| | cell | what it does | time |
|---|---|---|---|
| **A** | setup | GPU, ASTRA, Drive | 2 min |
| **B** | settings | paths, helpers | 1 min |
| **C** | registry | `manifest.csv` + VICTRE `.loc` → `registry.csv`, without opening files; integrity check | ~1 min |
| **C2** | DELTA check | `volume_box_mm` in every file (is the DELTA patch complete), through the Drive API (does not lock the connection); right after C | ~45–60 min |
| **D** | test analysis | per phantom on the test split: baselines, detection, z alignment, noise, datafit | ~8–10 h |
| **E** | collect + check | CSV files and a check of the expected results | 2 min |

**D resumes where it stopped:** each phantom is written as `phantoms/<id>.json`, existing ones are skipped.
If the session drops, run **A → B → D** again. For two parallel tabs, `SESSION`/`N_SESSIONS`.

**First a trial:** run D with `TRIAL = 3`, run E, then run notebook `02`.
If everything comes out, set `TRIAL = 0` and run all of it (the trial phantoms are not recomputed).

Outputs: `/content/drive/MyDrive/New_DBT/VICTRE-PAIRED-v11_analysis/`
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
## B · Settings

`dataset_generate.py` and `dataset_test.py` are read from the generation folder on Drive,
so the analysis uses **the same code** that produced the data (geometry, SART, FBP, metrics).
""")
code(r"""
import os, sys, json, time, csv, collections
from pathlib import Path
import numpy as np

NB_DIR = Path("/content/drive/MyDrive/victre_paired/code")                     # generation scripts
DATA   = Path("/content/drive/MyDrive/New_DBT/VICTRE-PAIRED-v11")      # dataset
RES    = Path("/content/drive/MyDrive/New_DBT/VICTRE-PAIRED-v11_analysis")   # output of this notebook

SPLIT      = "test"   # split for the baseline/detection/z analysis
TRIAL     = 0        # >0: only this many phantoms (trial). 0: all
SESSION    = 0        # parallel tab: 0..N_SESSIONS-1
N_SESSIONS = 1

for d in (RES, RES/"phantoms", RES/"panel"):
    d.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(NB_DIR)); os.chdir(NB_DIR)
import dataset_generate as G
import dataset_test as T          # metrics(), noise_ratios(), DOSES
%matplotlib inline

SPLITS, DENS, DOSES = ("train", "val", "test"), G.DENSITIES, T.DOSES     # DOSES = ['50','25','10']
assert all(v == 19.5 for v in G.DELTA.values()), f"dataset_generate.py on Drive is old (before the DELTA patch): DELTA={G.DELTA}"

files = {s: sorted(p for p in (DATA/s).glob("*.npz") if ".partial" not in p.name) for s in SPLITS}
print({s: len(v) for s, v in files.items()}, "total", sum(map(len, files.values())))

# ------------------------------------------------------------------ helpers
R_CORE, R_BG = 4, 12        # SDNR half-widths of the core and the background (voxels), v9 protocol

def sdnr_at(vol, z, h, w):
    # (core mean - background mean) / background s.d. The background is a 25x25 square in the same slice
    # and CONTAINS the core; the same definition as v9 evaluate.py, comparisons at the same locations.
    z, h, w = int(round(z)), int(round(h)), int(round(w))
    if not (0 <= z < vol.shape[0] and 0 <= h < vol.shape[1] and 0 <= w < vol.shape[2]):
        return None
    core = vol[max(0, z-1):z+2, max(0, h-R_CORE):h+R_CORE+1, max(0, w-R_CORE):w+R_CORE+1]
    bg   = vol[z, max(0, h-R_BG):h+R_BG+1, max(0, w-R_BG):w+R_BG+1]
    s = float(bg.std())
    return None if s < 1e-12 else float((core.mean() - bg.mean()) / s)

R_LES = int(round(G.CFG.lesion_radius_mm / (G.CFG.native_pixel_mm * G.CFG.downsample)))
DZ = np.arange(-3, 4)

def contrast(sl, r, c):
    yy, xx = np.ogrid[:sl.shape[0], :sl.shape[1]]
    d2 = (yy-r)**2 + (xx-c)**2
    core, ring = d2 <= R_LES**2, (d2 > (2*R_LES)**2) & (d2 <= (3*R_LES)**2)
    if core.sum() < 5 or ring.sum() < 20: return np.nan
    return float(sl[core].mean() - sl[ring].mean())

def peak_offset(vol, z, r, c):
    # As in D2 of the generation notebook: true slice of the lesion minus the .loc Z, parabolic interpolation.
    S = vol.shape[0]
    if not (3 <= z < S-3): return np.nan
    y = np.array([contrast(vol[int(z+d)].astype(np.float64), r, c) for d in DZ])
    if not np.isfinite(y).all(): return np.nan
    k = int(np.argmax(y))
    if k in (0, len(y)-1): return float(DZ[k])
    a, b, cc = y[k-1], y[k], y[k+1]
    den = a - 2*b + cc
    sub = 0.5*(a-cc)/den if abs(den) > 1e-12 else 0.0
    return float(DZ[k] + np.clip(sub, -1, 1))

def corr_masked(a, b, m):
    x = a[m].astype(np.float64); y = b[m].astype(np.float64)
    x -= x.mean(); y -= y.mean()
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))

def datafit(geo, vol, proj, valid):
    # ||s A x - p|| / ||p|| on valid rays, s by least squares. Independent of any reference:
    # consistency of the volume with the measurements. For sart_full s ~ 1 is expected (mm^-1).
    ax = geo.project(np.ascontiguousarray(vol, np.float32))[valid].astype(np.float64)
    p = proj[valid].astype(np.float64)
    s = float(ax @ p / max(ax @ ax, 1e-12))
    return s, float(np.linalg.norm(s*ax - p) / max(np.linalg.norm(p), 1e-12))

import concurrent.futures as _cf
_LOADER = _cf.ThreadPoolExecutor(1)
def load_full(f, timeout=900):
    # A hanging Drive read does not fail, it waits. The read runs in a separate thread with a time
    # limit: if it hangs, the cell stops and says what to do (finished phantoms are saved).
    def _rd():
        with np.load(f) as z:
            return {k: z[k] for k in z.files}
    try:
        return _LOADER.submit(_rd).result(timeout=timeout)
    except _cf.TimeoutError:
        raise RuntimeError(f"{f.name}: Drive did not respond in {timeout} s (connection hanging). "
                           "Runtime > Restart session, then A, B, D; finished phantoms are skipped.")

print("ready. R_LES =", R_LES, "px")
""")

md(r"""
## C · Registry — the whole collection, **without opening files** (~1 min)

Opening the 2,993 `.npz` files one by one from Drive locked the Drive connection. All fields of the registry
already exist in two small sources:

- **`manifest.csv`** (cell G of the generation notebook; read from each file itself): density, split,
  signal, slices, lesion count, photons, electronic noise, `victre_fbp`, saturation, size.
- **VICTRE's `.loc` files** (DIDSR GitHub, 4 small downloads): the split into masses / calcifications / controls,
  with exactly the generator's `lesion_table` rule (same bounds, same type rule).

**Integrity check:** in every phantom the masses + calcifications counted from the `.loc` must equal the
`lesions` count that `manifest` read from the file. Also `manifest` ↔ the file list on Drive.
""")
code(r"""
import pandas as pd, urllib.request, tarfile, io
t0 = time.time()
man = pd.read_csv(DATA/"manifest.csv", dtype={"patient": str})
print(f"manifest: {len(man)} rows", flush=True)
on_drive = {(s, f.stem) for s in SPLITS for f in files[s]}
in_man = set(zip(man.split, man.patient))
assert on_drive == in_man, (f"manifest and Drive differ: only on Drive {len(on_drive - in_man)}, "
                            f"only in the manifest {len(in_man - on_drive)}; example {sorted(on_drive ^ in_man)[:5]}")

# .loc: the same rule as the generator's lesion_index + lesion_table
LOC_URL = "https://raw.githubusercontent.com/DIDSR/VICTRE/master/Locations/DBT-{}.tar.gz"
locrows = {}
for dn in DENS:
    raw = urllib.request.urlopen(LOC_URL.format(dn), timeout=120).read()
    with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
        for mem in tf.getmembers():
            nm = Path(mem.name).name
            if not (nm.startswith("roi_S") and nm.endswith(".loc")): continue
            seed = nm[7:-4].lstrip("-")
            txt = tf.extractfile(mem).read().decode().splitlines()
            locrows[seed] = np.array([[float(v) for v in ln.split()] for ln in txt if len(ln.split()) == 4],
                                     np.float32).reshape(-1, 4)
    print(f"  .loc {dn}: downloaded", flush=True)

def counts(pid, sp, S):
    rows_ = locrows.get(pid.lstrip("-"))
    if rows_ is None: return None
    nm = nc = nk = 0
    for x, y, z, num in rows_:
        r, c = x / G.CFG.downsample, y / G.CFG.downsample
        if 0 <= z < S and 0 <= r < G.CFG.vol_rows and 0 <= c < G.CFG.vol_cols:
            if not sp: nk += 1
            elif num < 4: nc += 1
            else: nm += 1
    return nm, nc, nk

rows, no_loc, mismatch = [], [], []
for m_ in man.itertuples(index=False):
    sp = str(m_.signal_present).strip().lower() in ("true", "1")        # "True"/"False" in the csv
    cnt = counts(m_.patient, sp, int(m_.slices))
    if cnt is None: no_loc.append(m_.patient); continue
    nm, nc, nk = cnt
    if nm + nc != int(m_.lesions): mismatch.append((m_.patient, nm + nc, int(m_.lesions)))
    rows.append(dict(patient=m_.patient, split=m_.split, density=m_.density, signal_present=int(sp),
                     slices=int(m_.slices), n_mass=nm, n_calc=nc, n_control=nk,
                     photons=float(m_.photons_per_pixel), sigma_e=float(m_.electronic_sigma),
                     victre_fbp=int(m_.victre_fbp), victre_fbp_saturated=float(m_.victre_fbp_saturated),
                     size_mb=float(m_.size_mb)))
assert not no_loc, f"{len(no_loc)} phantoms without a .loc: {no_loc[:5]}"
assert not mismatch, f"the lesion count from the .loc differs from the file in {len(mismatch)} phantoms: {mismatch[:5]}"
order = {k: i for i, k in enumerate((s, f.stem) for s in SPLITS for f in files[s])}
rows.sort(key=lambda r: order[(r["split"], r["patient"])])

with open(RES/"registry.csv", "w", newline="") as h:
    w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

print(f"\n{len(rows)} phantoms, {sum(r['size_mb'] for r in rows)/1000:.1f} GB, {time.time()-t0:.0f} s")
print("integrity: manifest = Drive file list; .loc masses+calcifications = lesion count in the file, in every phantom -> OK")
print(f"{'density':<11}{'n':>6}{'SP':>6}{'SA':>6}{'mass':>7}{'calc':>7}{'control':>8}{'vfbp':>6}{'tr/va/te':>14}")
for dn in DENS + ("ALL",):
    R = [r for r in rows if dn == "ALL" or r["density"] == dn]
    sp_ = collections.Counter(r["split"] for r in R)
    print(f"{dn:<11}{len(R):>6}{sum(r['signal_present'] for r in R):>6}{sum(1-r['signal_present'] for r in R):>6}"
          f"{sum(r['n_mass'] for r in R):>7}{sum(r['n_calc'] for r in R):>7}{sum(r['n_control'] for r in R):>8}"
          f"{sum(r['victre_fbp'] for r in R):>6}{sp_['train']:>6}/{sp_['val']}/{sp_['test']}")
""")

md(r"""
### C2 · DELTA check — every file, **without the Drive connection (FUSE)** (~45–60 min)

`DELTA` is 19.5 in all four densities (fatty at generation, the others patched by `07_delta_patch`).
To find any file left with the old value, **one small field** (`volume_box_mm`) is read from every file.
D also checks the test phantoms; this cell covers train/val as well.

The Drive connection (`/content/drive`) locks after ~20 files when files are opened in a row.
This cell reads the files not from there but **through the Drive API with byte-range requests**: from each file
only the zip directory and that small field (a few KB) are downloaded; every request has a 60 s timeout and cannot hang.
On the first run Google opens a **permission window** (read only) → accept it.
Resumes where it stopped (`delta_parts_v107.jsonl`). Since it does not use the Drive connection, it is also safe to run before D.
""")
code(r"""
import io, zipfile
import pandas as pd
from google.colab import auth
auth.authenticate_user()
import google.auth
from google.auth.transport.requests import AuthorizedSession
creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/drive.readonly"])
API_S = AuthorizedSession(creds)
API = "https://www.googleapis.com/drive/v3/files"
FOLDER, SHORTCUT = "application/vnd.google-apps.folder", "application/vnd.google-apps.shortcut"
COMMON = dict(supportsAllDrives=True, includeItemsFromAllDrives=True)

def api_get(url, tries=4, **kw):
    for t in range(tries):
        try:
            r = API_S.get(url, timeout=60, **kw); r.raise_for_status(); return r
        except Exception as e:
            if t == tries - 1: raise
            time.sleep(5 * (t + 1))

def ls(parent, extra=""):
    out, tok = [], None
    while True:
        j = api_get(API, params=dict(q=f"'{parent}' in parents and trashed=false{extra}", pageSize=1000,
                                     fields="nextPageToken,files(id,name,size,mimeType,shortcutDetails)",
                                     pageToken=tok, **COMMON)).json()
        out += j.get("files", []); tok = j.get("nextPageToken")
        if not tok: return out

def child_folder(parent, name):
    c = [x for x in ls(parent, " and name='" + name.replace("'", "\\'") + "'") if x["mimeType"] in (FOLDER, SHORTCUT)]
    assert len(c) == 1, f"folder '{name}' found {len(c)} times on Drive"
    x = c[0]
    return x["shortcutDetails"]["targetId"] if x["mimeType"] == SHORTCUT else x["id"]

fid = "root"
for part in DATA.relative_to("/content/drive/MyDrive").parts:
    fid = child_folder(fid, part)
api_files = {}
for s in SPLITS:
    for x in ls(child_folder(fid, s), " and name contains '.npz'"):
        if ".partial" not in x["name"]:
            api_files[x["name"][:-4]] = (x["id"], int(x["size"]))
print(f"Drive API: {len(api_files)} files listed ({sum(map(len, files.values()))} in the FUSE listing)", flush=True)

class RangeFile(io.RawIOBase):
    # Presents a file on Drive as a readable file by downloading only the requested byte ranges.
    def __init__(self, fid, size): self.fid, self.size, self.pos = fid, size, 0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off); return self.pos
    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0: return 0
        d = api_get(f"{API}/{self.fid}", params=dict(alt="media", **COMMON),
                    headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"}).content[:n]
        b[:len(d)] = d; self.pos += len(d); return len(d)

def read_box(stem):
    fid_, size = api_files[stem]
    with zipfile.ZipFile(io.BufferedReader(RangeFile(fid_, size), buffer_size=1 << 16)) as zf, \
         zf.open("volume_box_mm.npy") as h:
        return np.lib.format.read_array(h)

PARTS_LOG = RES/"delta_parts_v107.jsonl"      # after the DELTA patch; the earlier delta_parts.jsonl covered fatty only
done = {}
if PARTS_LOG.exists():
    for ln in PARTS_LOG.read_text().splitlines():
        try: r = json.loads(ln); done[r["patient"]] = r
        except Exception: pass
man = pd.read_csv(DATA/"manifest.csv", dtype={"patient": str})
DN_OF = dict(zip(man.patient, man.density))
fatset = set(man.patient)                     # (historical name: now all files)
fat = [(s, f) for s in SPLITS for f in files[s] if f.stem in fatset and f.stem not in done]
missing = [f.stem for s, f in fat if f.stem not in api_files]
assert not missing, f"{len(missing)} files not found through the Drive API: {missing[:5]}"
Z0 = {dn: G.volume_box(dn, 1)[4] for dn in DENS}   # z0 = DELTA - (SDD-SID), independent of the slice count
print(f"{len(done)} checked before, {len(fat)} left; expected z0 = {Z0}", flush=True)

t0 = time.time()
for i, (s, f) in enumerate(fat, 1):
    box = read_box(f.stem)
    r = dict(patient=f.stem, split=s, density=DN_OF[f.stem], z0_mm=round(float(box[4]), 4),
             z0_ok=int(abs(float(box[4]) - Z0[DN_OF[f.stem]]) < 1e-3))
    with open(PARTS_LOG, "a") as h: h.write(json.dumps(r) + "\n")
    done[f.stem] = r
    if i <= 3 or i % 10 == 0 or i == len(fat):
        el = time.time() - t0
        print(f"  {i}/{len(fat)}  {el/i:.1f} s/file  | ~{el/i*(len(fat)-i)/60:.0f} min left", flush=True)

bad = [r["patient"] for r in done.values() if not r["z0_ok"]]
print(f"\nDELTA check ({len(done)}/{len(fatset)} files):", "CLEAN" if not bad else
      f"{len(bad)} FILES WITH THE OLD DELTA  <-- STOP, complete 07_delta_patch: {bad[:10]}")
""")

md(r"""
## D · Test analysis — per phantom

For each test phantom, the same reconstructions as `dataset_test.py`
(full-dose FBP; OS-SART and FBP at each dose), and in addition:

- **agreement:** PSNR, SSIM, MSE (`dataset_test.metrics`, same scale matching and window) and masked correlation,
  against `sart_full` and — where present — `victre_fbp` (the latter **reported only**)
- **detection:** SDNR at every lesion and control location (v9 protocol), in every volume
- **z alignment:** peak offset relative to the `.loc` Z for every mass and calcification (D2 of the generation notebook)
- **noise:** `dataset_test.noise_ratios` (measured / predicted extra variance)
- **datafit:** `‖sAx − p‖/‖p‖`; for `sart_full`, `victre_fbp` and full-dose FBP

Also, for one phantom with lesions per density, figure panels are saved as `panel/<density>.npz`
(so the figure notebook needs no GPU). Phantoms with `victre_fbp` are preferred.

~1.5–2 min per phantom. If interrupted, run again; finished phantoms are skipped.
""")
code(r"""
work = [f for i, f in enumerate(files[SPLIT]) if i % N_SESSIONS == SESSION]
if TRIAL: work = work[:TRIAL]
todo = [f for f in work if not (RES/"phantoms"/f"{f.stem}.json").exists()]
print(f"{SPLIT}: {len(work)} phantoms in this tab, {len(work)-len(todo)} finished, {len(todo)} left\n", flush=True)

def want_panel(dn, sp, has_vf, les):
    if not sp or not (les[:, 4] == 1).any(): return False
    p = RES/"panel"/f"{dn}.npz"
    if not p.exists(): return True
    with np.load(p) as old:                      # a panel without vfbp is replaced when one with vfbp arrives
        return has_vf and not bool(old["victre_fbp_present"])

t_all = time.time()
for i, f in enumerate(todo):
    t = time.time()
    print(f"[{i+1}/{len(todo)}] {f.stem} reading ...", end=" ", flush=True)
    d = load_full(f)
    dn, gt, m = str(d["density"]), d["sart_full"], d["breast_mask"]
    S, les, valid = gt.shape[0], d["lesions"], d["proj_valid"]
    geo = G.Geometry(dn, S)
    pf = d["proj_full"].astype(np.float32)

    vols = {("FBP", "100"): geo.fbp(pf)}
    for k in DOSES:
        p = d[f"proj_{k}"].astype(np.float32)
        vols[("SART", k)] = geo.os_sart(p, valid)
        vols[("FBP", k)] = geo.fbp(p)

    has_vf = bool(d["victre_fbp_present"])
    vf = (d["victre_fbp"].astype(np.float32) / 65535.0 * float(d["victre_fbp_scale"])) if has_vf else None
    refs = {"sart_full": gt, **({"victre_fbp": vf} if has_vf else {})}

    rec = dict(patient=f.stem, split=SPLIT, density=dn, signal_present=int(bool(d["signal_present"])),
               victre_fbp_present=int(has_vf), slices=S, metrics=[], sdnr=[], z=[], datafit=[],
               z0_ok=int(abs(float(d["volume_box_mm"][4]) - G.volume_box(dn, S)[4]) < 1e-3))   # DELTA check

    # --- agreement
    for (meth, dose), v in vols.items():
        for rn, r in refs.items():
            mm = T.metrics(v, r, m)
            rec["metrics"].append(dict(method=meth, dose=dose, ref=rn, psnr=float(mm["psnr"]),
                                       ssim=float(mm["ssim"]), mse=float(mm["mse"]), corr=corr_masked(v, r, m)))
    if has_vf:
        mm = T.metrics(gt, vf, m)
        rec["metrics"].append(dict(method="GT", dose="100", ref="victre_fbp", psnr=float(mm["psnr"]),
                                   ssim=float(mm["ssim"]), mse=float(mm["mse"]), corr=corr_masked(gt, vf, m)))

    # --- detection: every location, every volume. A location counts only if it is defined in the GT and in ALL
    # reconstructions: the methods are compared at the same locations (v9 protocol). victre_fbp
    # exists only in the phantoms that carry it, so it is not part of this condition.
    allv = {("GT", "100"): gt, **vols}
    for s_, r_, c_, num, kind in les:
        xs = {k: sdnr_at(v, s_, r_, c_) for k, v in allv.items()}
        if any(x is None for x in xs.values()):
            continue
        if has_vf:
            xv = sdnr_at(vf, s_, r_, c_)
            if xv is not None: xs[("VFBP", "100")] = xv
        for (meth, dose), x in xs.items():
            rec["sdnr"].append(dict(method=meth, dose=dose, kind=int(kind), num=int(num), sdnr=x))

    # --- z alignment
    for s_, r_, c_, num, kind in les:
        if int(kind) not in (0, 1): continue
        for vn, v in (("sart_full", gt), ("victre_fbp", vf)):
            if v is None: continue
            o = peak_offset(v, s_, r_, c_)
            if np.isfinite(o):
                rec["z"].append(dict(volume=vn, kind=int(kind), s=float(s_), zfrac=float(s_)/S, offset=o))

    # --- datafit
    for vn, v in (("sart_full", gt), ("victre_fbp", vf), ("FBP_100", vols[("FBP", "100")])):
        if v is None: continue
        sc, df = datafit(geo, v, pf, valid)
        rec["datafit"].append(dict(volume=vn, scale=sc, datafit=df))

    # --- noise
    rec["noise"] = {k: float(v) for k, v in T.noise_ratios(d).items()}

    # --- figure panel
    sp = bool(d["signal_present"])
    if want_panel(dn, sp, has_vf, les):
        zp = int(round(float(les[les[:, 4] == 1][0, 0])))
        np.savez_compressed(
            RES/"panel"/f"{dn}.npz", patient=f.stem, density=dn, z=zp, slices=S, mask=m[zp], lesions=les,
            sart_full=gt[zp], victre_fbp_present=has_vf,
            victre_fbp=(vf[zp] if has_vf else np.zeros((0, 0), np.float32)),
            **{f"rec_{meth}_{dose}": v[zp].astype(np.float32) for (meth, dose), v in vols.items()},
            proj_full=d["proj_full"][12], proj_valid=d["proj_valid"][12],
            **{f"proj_{k}": d[f"proj_{k}"][12] for k in DOSES},
            dm_full=d["dm_full"], dm_valid=d["dm_valid"], **{f"dm_{k}": d[f"dm_{k}"] for k in DOSES})

    rec["seconds"] = round(time.time() - t, 1)
    tmp = RES/"phantoms"/f"{f.stem}.json.tmp"
    tmp.write_text(json.dumps(rec, default=float))
    tmp.rename(RES/"phantoms"/f"{f.stem}.json")

    el = time.time() - t_all
    eta = el / (i+1) * (len(todo) - i - 1) / 3600
    print(f"{dn:<9} vfbp={int(has_vf)}  {rec['seconds']:.0f} s  | ~{eta:.1f} h left", flush=True)

print(f"\ndone: {len(todo)} phantoms, {(time.time()-t_all)/3600:.2f} h")
""")

md(r"""
## E · Collect and check

`phantoms/*.json` → five CSV files. Then the expected results of the development release and the z alignment
are checked. **If a line says "CHECK", find the cause before going on to the figures.**
""")
code(r"""
import pandas as pd
recs = [json.loads(p.read_text()) for p in sorted((RES/"phantoms").glob("*.json"))]
assert recs, "no phantoms -> run D first"
BASE = ["patient", "split", "density", "signal_present", "victre_fbp_present"]
def flat(key):
    return pd.DataFrame([{**{k: r[k] for k in BASE}, **x} for r in recs for x in r[key]])

B, P, Z, DF = flat("metrics"), flat("sdnr"), flat("z"), flat("datafit")
N = pd.DataFrame([{**{k: r[k] for k in BASE}, "key": k, "ratio": v} for r in recs for k, v in r["noise"].items()])
for name, df in (("baseline", B), ("sdnr", P), ("zalign", Z), ("datafit", DF), ("noise", N)):
    df.to_csv(RES/f"{name}.csv", index=False)
    print(f"{name+'.csv':<14} {len(df):>7} rows")
print(f"\n{len(recs)} phantoms, mean {np.mean([r['seconds'] for r in recs]):.0f} s/phantom")

def ok(c): return "OK" if c else "CHECK  <--"
print("\n--- expected results (development release) ---")
g = B[B.ref == "sart_full"].groupby(["method", "dose"]).psnr.mean()
for k in DOSES:
    print(f"SART {k:>2}%  PSNR {g[('SART', k)]:6.2f} dB")
print(f"SART 25% ~27.8 dB (DBT 2026v1: 27.83)        -> {ok(abs(g[('SART','25')] - 27.8) < 1.5)}")
print(f"SART 50 > 25 > 10                             -> {ok(g[('SART','50')] > g[('SART','25')] > g[('SART','10')])}")
print(f"FBP ~18 dB at every dose  {[round(float(g[('FBP', k)]), 1) for k in ('100',) + tuple(DOSES)]} "
      f"-> {ok(all(abs(g[('FBP', k)] - 18) < 3 for k in ('100',) + tuple(DOSES)))}")
nm = N.groupby("key").ratio.median()
print(f"noise ratios (median) {nm.round(3).to_dict()}  -> {ok(nm.between(0.85, 1.15).all())}")
vc = B[(B.method == "GT") & (B.ref == "victre_fbp")]
if len(vc):
    print(f"corr(victre_fbp, sart_full) min {vc['corr'].min():.3f}, by density "
          f"{vc.groupby('density')['corr'].mean().round(3).to_dict()}  -> {ok(vc['corr'].min() >= 0.15)}")

print("\n--- z alignment (masses, offset in slices; |mean| < 0.3 expected) ---")
zm = Z[Z.kind == 1]
for vn in ("sart_full", "victre_fbp"):
    for dn in DENS:
        v = zm[(zm.volume == vn) & (zm.density == dn)].offset
        if len(v) >= 5:
            # sart_full: DELTA is our calibration -> checked. victre_fbp: VICTRE's own volume, its z
            # is stored untouched (load_victre_fbp pools only in plane) -> reported only.
            print(f"{vn:<11} {dn:<10} n={len(v):>4}  {v.mean():+.2f} +- {v.std(ddof=1)/np.sqrt(len(v)):.2f}"
                  f"  -> {ok(abs(v.mean()) < 0.3) if vn == 'sart_full' else 'reported (VICTRE volume)'}")

print("\n--- detection: control SDNR in the GT should be ~0 ---")
gt = P[(P.method == "GT")]
for kind, name in ((1, "mass"), (0, "calcification"), (-1, "control")):
    v = gt[gt.kind == kind].sdnr
    print(f"{name:<14} n={len(v):>5}  mean {v.mean():+.3f}")

zb = [r["patient"] for r in recs if not r.get("z0_ok", 1)]
print(f"\nDELTA check, test ({len(recs)} phantoms): " + ("CLEAN  -> OK" if not zb else f"{len(zb)} OLD DELTA  CHECK  <-- {zb[:5]}"))

print("\n--- datafit ---")
print(DF.groupby(["volume", "density"]).datafit.median().unstack().round(4))
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
