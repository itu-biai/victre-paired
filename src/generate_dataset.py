"""
generate_dataset.py — build the VICTRE-PAIRED v6 dataset from VICTRE source data.

For every eligible VICTRE patient this script produces a paired sample:
  * clean_proj        — the 25 limited-angle Monte-Carlo projections, flat-field
                        corrected, penumbra-cleaned, downsampled, normalized and
                        density-aligned;
  * noisy_proj{,_full,_quarter} — Poisson-Gaussian noisy projections at three
                        dose levels (half dose is the default `noisy_proj`);
  * clean, noisy, mask — the paired FBP reconstruction volume (reference),
                        a noisy reconstruction, and the breast mask;
  * lesion_coords     — phantom-accurate lesion locations mapped into the
                        reconstruction grid, with lesion type;
  * per-sample metadata (seed, density, native_z, scale factors, sigmas ...).

Samples are written in chunks of 8 patients as compressed .npz files, split
deterministically into train/val/test.

USAGE (designed for parallel execution across five Google Colab sessions):
    Set RUN_MODE to one of {"train_1","train_2","train_3","val","test"} and run.
    The train split is produced in three parallel shards. Completed chunks are
    skipped, so an interrupted run can simply be restarted.

INPUTS (expected under a Google Drive mount; adapt paths as needed):
    VICTRE/projections/<seed>/<series>/*.dcm     raw MC projection DICOMs
    VICTRE/dicoms_v2/<uid>/*.dcm                  FBP reconstruction DICOMs
    VICTRE/dicoms_v2/manifest.csv                 SEED <-> series UID manifest
    ff_poly2_25.npy                               estimated flat-field (see paper)
    Lesion .loc files are fetched from github.com/DIDSR/VICTRE on first run.

This is the exact production logic used for the released dataset; only comments
and messages have been translated and documentation added.
"""
import os
import re
import glob
import time
import random
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import pydicom
from scipy.ndimage import binary_dilation

from constants import (
    DS, RECON_Z as ZOUT, NA, RECON_H as Hr, RECON_W as Wr,
    PROJ_ROWS as Hp, PROJ_COLS as Wp, VALLEY, DILATE, OFFSET,
    DOSES, S_ELEC, SEED_SPLIT, CHUNK, SPLIT, fmt_seconds as fmt,
)

# ── 🚨 CHANGE THIS IN EACH PARALLEL SESSION ────────────────────────────────
RUN_MODE = "train_1"    # one of: "train_1" | "train_2" | "train_3" | "val" | "test"
# ───────────────────────────────────────────────────────────────────────────

ROOT = Path("/content/drive/MyDrive/New_DBT/VICTRE")
PROJD, REKON = ROOT / "projections", ROOT / "dicoms_v2"
OUT = Path("/content/drive/MyDrive/New_DBT/VICTRE-PAIRED-v6")
LOCAL = Path("/content/v6"); LOCAL.mkdir(exist_ok=True)
REPO, LOCEXT = "/content/VICTRE_repo", "/content/VICTRE_repo/Locations/ext"
FF2 = "/content/drive/MyDrive/New_DBT/ff_poly2_25.npy"


def key_of(s):
    """Canonical patient key: SEED string without a leading sign."""
    return str(s).lstrip('-')


# ── PREPARATION: fetch lesion .loc files + check flat-field ────────────────
print(f"[{RUN_MODE}] Preparing...", flush=True)
if not os.path.isdir(LOCEXT) or not glob.glob(f"{LOCEXT}/**/*.loc", recursive=True):
    print("  .loc missing -> downloading from GitHub...", flush=True)
    if not os.path.isdir(REPO):
        subprocess.run(f"git clone --depth 1 https://github.com/DIDSR/VICTRE.git {REPO}",
                       shell=True, check=True)
    os.makedirs(LOCEXT, exist_ok=True)
    for f in glob.glob(f"{REPO}/Locations/DBT-*.tar.gz"):
        subprocess.run(f'tar -xzf "{f}" -C "{LOCEXT}"', shell=True, check=True)
nloc = len(glob.glob(f"{LOCEXT}/**/*.loc", recursive=True))
print(f"  .loc files: {nloc} {'OK' if nloc == 2994 else 'WARN'}", flush=True)
assert nloc > 2900, "ERROR: lesion .loc files missing"
assert os.path.exists(FF2), f"ERROR: flat-field not found: {FF2}"
ff2 = np.load(FF2)
print(f"  flat-field: {ff2.shape} OK", flush=True)

# ── SEED maps (SEED -> lesion file / signal-present / density / recon UID) ──
s2loc, s2pos, s2dens = {}, {}, {}
for p in glob.glob(f"{LOCEXT}/**/*.loc", recursive=True):
    m = re.match(r'roi_(SP|SA)_(-?\d+)\.loc', os.path.basename(p))
    if m:
        k = key_of(m.group(2))
        s2loc[k] = p
        s2pos[k] = (m.group(1) == 'SP')          # SP = signal-present, SA = signal-absent
        s2dens[k] = p.split('/DBT-')[1].split('/')[0]
man = pd.read_csv(str(REKON / "manifest.csv"))
man["_k"] = man["PatientID"].astype(str).map(key_of)
rm = man[man["SeriesDescription"].astype(str).str.contains("slice|DBT|recon", case=False, na=False)]
s2uid = dict(zip(rm["_k"], rm["SeriesInstanceUID"].astype(str)))
print(f"  SEEDs: {len(s2loc)} | signal-present={sum(s2pos.values())} | recon UIDs={len(s2uid)}\n",
      flush=True)


# ── HELPER FUNCTIONS ───────────────────────────────────────────────────────
def ds2(x):
    """4x average-pool spatial downsampling of a 2-D array."""
    return F.avg_pool2d(torch.from_numpy(x)[None, None].float(), DS)[0, 0].numpy()


def fit(a, TH, TW, pad=0.0):
    """Crop or pad the last two dims of `a` to exactly (TH, TW)."""
    a = a[..., :min(a.shape[-2], TH), :min(a.shape[-1], TW)]
    h, w = a.shape[-2:]
    if h < TH or w < TW:
        a = np.pad(a, [(0, 0)] * (a.ndim - 2) + [(0, TH - h), (0, TW - w)], constant_values=pad)
    return a


def shift2d(img, dr, dc):
    """Integer-pixel shift of a 2-D image by (dr, dc) with zero fill.
    Used instead of np.roll so content does not wrap around the edges."""
    o = np.zeros_like(img)
    h, w = img.shape
    yd = slice(max(0, dr), h - max(0, -dr)); ys = slice(max(0, -dr), h - max(0, dr))
    xd = slice(max(0, dc), w - max(0, -dc)); xs = slice(max(0, -dc), w - max(0, dc))
    o[yd, xd] = img[ys, xs]
    return o


def load_proj(seed, dens):
    """Load and process the 25 MC projections for one patient.

    Steps: read DICOMs in acquisition order; convert to attenuation via
    p = log(flatfield) - log(intensity); restrict to the air bounding box;
    remove the penumbra strip (attenuation above VALLEY); downsample; normalize
    by the 99.8th percentile; crop/pad to (25, Hp, Wp); density-align each view.
    Returns (projections, normalization_scale) or (None, None) on failure.
    """
    fs = sorted(glob.glob(f'{PROJD}/{seed}/*/*.dcm'),
                key=lambda p: int(pydicom.dcmread(p, stop_before_pixels=True).InstanceNumber))
    if len(fs) != NA:
        return None, None
    out = []
    for a, f in enumerate(fs):
        I = pydicom.dcmread(f).pixel_array.astype(np.float32)
        air = I > 5000
        ys, xs = np.where(air)
        if len(ys) == 0:
            return None, None
        m = np.zeros_like(air)
        m[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
        p = np.clip(np.log(ff2[a]) - np.log(np.clip(I, 1, None)), 0, None)
        p = np.where(m, p, 0.0)
        bad = binary_dilation(p > VALLEY, iterations=DILATE)          # penumbra strip
        out.append(ds2(np.where(bad, 0.0, p)))
    out = np.stack(out)
    nz = out[out > 0]
    if nz.size == 0:
        return None, None
    q = float(np.percentile(nz, 99.8) + 1e-6)
    out = fit(np.clip(out / q, 0, 1), Hp, Wp)
    sl, ic, co = OFFSET.get(dens, OFFSET["scattered"])
    aligned = np.stack([shift2d(out[a], int(round(sl * a + ic)), co) for a in range(NA)])
    return aligned.astype(np.float32), q


def load_recon(uid):
    """Load the FBP reconstruction volume for one series UID.

    Reads slices in InstanceNumber order, downsamples in-plane, resamples the
    slice axis to ZOUT via trilinear interpolation, normalizes by the 99.5th
    percentile and crops/pads to (ZOUT, Hr, Wr). Returns (volume, native_z, scale).
    """
    fs = list((REKON / uid).glob("*.dcm"))
    res = sorted([(int(pydicom.dcmread(str(f), stop_before_pixels=True).InstanceNumber),
                   pydicom.dcmread(str(f)).pixel_array.astype(np.float32)) for f in fs],
                 key=lambda x: x[0])
    vol = np.stack([ds2(I) for _, I in res])
    nz = len(res)
    v = torch.from_numpy(vol)[None, None]
    v = F.interpolate(v, size=(ZOUT, vol.shape[1], vol.shape[2]),
                      mode="trilinear", align_corners=False)[0, 0].numpy()
    q = float(np.percentile(v, 99.5) + 1e-3)
    return fit(np.clip(v / q, 0, 1).astype(np.float32), Hr, Wr), nz, q


def load_lesions(k, nz):
    """Map phantom lesion locations into the reconstruction grid.

    VICTRE .loc gives (X, Y, Z, type) in native coordinates; here X,Y are
    downsampled by DS and Z is rescaled from the native slice count to ZOUT.
    Only signal-present patients have lesions. Returns an (n, 4) array
    [z, h, w, type], empty if none in range.
    """
    p = s2loc.get(k)
    if p is None or not s2pos.get(k, False):
        return np.zeros((0, 4), np.float32)
    loc = np.loadtxt(p)
    if loc.size == 0:
        return np.zeros((0, 4), np.float32)
    if loc.ndim == 1:
        loc = loc[None]
    o = [[Z * ZOUT / nz, X / DS, Y / DS, t] for X, Y, Z, t in loc
         if 0 <= Z * ZOUT / nz < ZOUT and 0 <= X / DS < Hr and 0 <= Y / DS < Wr]
    return np.array(o, np.float32) if o else np.zeros((0, 4), np.float32)


def add_noise(arr, a, s, rng):
    """Apply the Poisson-Gaussian dose model at gain `a` and electronic std `s`.
    Returns the noisy array (clipped to [0,1]) and the measured masked noise std."""
    o = rng.poisson(np.maximum(arr / (a + 1e-8), 1e-6)).astype(np.float32) * a \
        + rng.standard_normal(arr.shape).astype(np.float32) * s
    o = np.clip(o, 0, 1).astype(np.float32)
    m = arr > 0
    return o, float((o[m] - arr[m]).std()) if m.any() else float(s)


# ── SCAN: find eligible patients (metadata-only, parallel I/O) ─────────────
print(f"[{RUN_MODE}] Scanning (metadata-only)...", flush=True)
pdirs = sorted([d for d in PROJD.iterdir() if d.is_dir()])


def scan(pd_):
    """Return (key, recon_uid) if the patient has 25 projections and a
    reconstruction series present on disk, else None."""
    try:
        k = key_of(pd_.name)
        if k not in s2loc:
            return None
        uid = s2uid.get(k)
        if uid is None or not (REKON / uid).exists():
            return None
        sers = [d for d in pd_.iterdir() if d.is_dir()]
        if not sers or len(list(sers[0].glob("*.dcm"))) != NA:
            return None
        if not list((REKON / uid).glob("*.dcm")):
            return None
        return (k, uid)
    except Exception:
        return None


res = [None] * len(pdirs)
t0 = time.time()
dn = 0
with ThreadPoolExecutor(max_workers=16) as ex:
    fut = {ex.submit(scan, pdirs[i]): i for i in range(len(pdirs))}
    for f in as_completed(fut):
        res[fut[f]] = f.result()
        dn += 1
        if dn % 500 == 0:
            print(f"  {dn}/{len(pdirs)} | {fmt(time.time() - t0)}", flush=True)
pts = [r for r in res if r]
print(f"[{RUN_MODE}] eligible patients: {len(pts)} (2761 expected)", flush=True)
assert len(pts) > 2700, f"WARN: too few ({len(pts)}) — stopping"

# ── SPLIT (deterministic; train produced in three shards) ──────────────────
random.Random(SEED_SPLIT).shuffle(pts)
n = len(pts)
nt = int(n * SPLIT["train"]); nv = int(n * SPLIT["val"])
tr = pts[:nt]; ps = len(tr) // 3
TASKS = {"train_1": ("train", tr[:ps]), "train_2": ("train", tr[ps:2 * ps]),
         "train_3": ("train", tr[2 * ps:]), "val": ("val", pts[nt:nt + nv]),
         "test": ("test", pts[nt + nv:])}
split_name, items = TASKS[RUN_MODE]
print(f"[{RUN_MODE}] {len(items)} patients to process\n", flush=True)

# ── GENERATION ─────────────────────────────────────────────────────────────
od = OUT / split_name
od.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED_SPLIT)
buf = []; ci = 0; done = 0; skip = 0; t0 = time.time()


def flush(buf, ci):
    """Write one chunk of up to CHUNK patients as a compressed .npz.
    Data is written locally first, then copied to Drive (FUSE-safe)."""
    fn = f"{RUN_MODE}_chunk_{ci:05d}.npz"
    lp = LOCAL / fn
    d = dict(
        clean=np.stack([b["cv"] for b in buf]), noisy=np.stack([b["nv"] for b in buf]),
        mask=np.stack([b["mk"] for b in buf]), is_pos=np.array([b["ip"] for b in buf], bool),
        clean_proj=np.stack([b["cp"] for b in buf]),
        seed=np.array([b["sd"] for b in buf]), density=np.array([b["dn"] for b in buf]),
        native_z=np.array([b["nz"] for b in buf], np.int16),
        lesion_coords=np.stack([b["les"] for b in buf]),
        lesion_count=np.array([b["nl"] for b in buf], np.int16),
        recon_scale=np.array([b["rq"] for b in buf], np.float32),
        proj_scale=np.array([b["pq"] for b in buf], np.float32),
        sigma_recon=np.array([b["sr"] for b in buf], np.float32),
        dose_levels=np.array([1.0, 0.5, 0.25], np.float32),
        dose_gains=np.array([DOSES["full"], DOSES["half"], DOSES["quarter"]], np.float32),
        elec_noise=np.float32(S_ELEC), strip_valley=np.float32(VALLEY))
    for tag in DOSES:
        key = "noisy_proj" if tag == "half" else f"noisy_proj_{tag}"
        d[key] = np.stack([b["np_"][tag] for b in buf])
        skey = "sigma" if tag == "half" else f"sigma_{tag}"
        d[skey] = np.array([b["sg"][tag] for b in buf], np.float32)
    np.savez_compressed(str(lp), **d)
    shutil.copy2(str(lp), str(od / fn))
    os.remove(str(lp))


for k, uid in items:
    if (od / f"{RUN_MODE}_chunk_{ci:05d}.npz").exists() and not buf:
        ci += 1; done += CHUNK; continue                     # skip an already-complete chunk
    try:
        dens = s2dens.get(k, "?")
        cp, pq = load_proj(k, dens)
        if cp is None:
            skip += 1; continue
        cv, nz, rq = load_recon(uid)
        mk = (cv > 0.08).astype(np.float32)
        nv, sr = add_noise(cv, DOSES["half"], S_ELEC, rng)
        np_, sg = {}, {}
        for tag, a in DOSES.items():
            np_[tag], sg[tag] = add_noise(cp, a, S_ELEC, rng)
        les = load_lesions(k, nz)
        lp_ = np.zeros((8, 4), np.float32); nl = min(len(les), 8)
        if nl:
            lp_[:nl] = les[:nl]
        buf.append(dict(cv=cv, nv=nv, mk=mk, ip=s2pos.get(k, False), cp=cp, np_=np_, sg=sg,
                        sd=int(k), dn=dens, nz=nz, les=lp_, nl=nl, rq=rq, pq=pq, sr=sr))
        done += 1
    except Exception as e:
        print(f"  error ({k}): {e}", flush=True); skip += 1; continue
    if len(buf) >= CHUNK:
        flush(buf, ci); buf = []; ci += 1
        el = time.time() - t0
        print(f"  [{RUN_MODE}] chunk {ci} | {done}/{len(items)} patients | skipped {skip} | "
              f"{fmt(el)} | ETA ~{fmt(el / done * (len(items) - done))}", flush=True)
if buf:
    flush(buf, ci); ci += 1

print(f"\n[{RUN_MODE}] DONE: {ci} chunks, {done} patients, {skip} skipped, {fmt(time.time() - t0)}",
      flush=True)
