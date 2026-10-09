#!/usr/bin/env python3
"""
VICTRE-Paired — dataset generation (data format v9).

This is the script that produced the released dataset. What changed relative to the
previous format, and the measurement behind each change: CHANGELOG_v9.md.

Builds paired projection/reconstruction chunks from the VICTRE in-silico trial
source data (Badano et al., 2018;
https://www.cancerimagingarchive.net/collection/victre/).

This script assumes the VICTRE source data is already on disk:
  - `projections/<SEED>/<UID>/*.dcm`   25 raw Monte-Carlo projection DICOMs/patient
  - `dicoms_v2/<UID>/*.dcm`            the corresponding FBP reconstruction slices
  - `manifest.csv`                     SEED -> SeriesInstanceUID mapping
  - lesion `.loc` files                from https://github.com/DIDSR/VICTRE (Locations/)
  - a flat-field estimate              `flatfield/ff_base.npy`,
                                        `flatfield/coefficients.json` (see README,
                                        "Flat-field"). `valley.json` is no longer
                                        read. The true VICTRE flat-field is
                                        not published; one is estimated per density
                                        class from air-only detector regions (paper,
                                        Methods).
  - split assignment                   `splits.csv` next to constants.py (seed,split);
                                        see README, "Split assignment".

Runs on CPU — no GPU/LEAP needed for generation (only for validate_dataset.py and
run_baselines.py). Safe to interrupt and resume: each chunk is written to a local
temp file, copied to the output directory under a temp name, then atomically
renamed; already-complete chunks are skipped on the next run.

Usage
-----
    python generate_dataset.py --split test --dry-run 3    # diagnostics, scratch folder
    python generate_dataset.py --split train --shard 0 --n-shards 3
    python generate_dataset.py --split val
    python generate_dataset.py --split test

Run the dry run first and read the diagnostics block: it reports whether the
illuminated bounding box is itself cutting tissue away, the highest attenuation now
kept, and that nothing lands on the coding ceiling.

For a large train split, shard the work across parallel processes/machines with
--shard i --n-shards N (every i-th patient, by sorted seed, modulo N). This is a
parallelism convenience only; split membership never depends on the shard count.
"""

import os, sys, glob, json, re, gc, time, argparse, shutil, zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn

try:
    import pydicom
except ImportError:
    raise SystemExit("pip install pydicom")

from constants import (ZOUT, TH, TW, NA, PH, PW, SID, SDD, DET_PIX, VOX_XY,
                       NATIVE_PIX, OFFX_C, OFFY_A, OFFY_B, DELTA, NATIVE_XY,
                       DOSES, DOSE_IDX, ELEC_PHOTONS, NOISY_LO, NOISY_HI,
                       MASK_THR, MASK_ANCHOR, FORMAT_VERSION, CHANGES_FROM_V8,
                       SPLIT_CSV)
from noise import add_noise, encode as encode_noisy, decode as decode_noisy
# ANG and the trajectory itself are used by geometry.py at reconstruction time,
# not during generation.

DS = 4              # native -> stored downsample factor
CHUNK = 8            # patients per .npz
AIR_THR = 5000       # raw detector counts above which a pixel is "illuminated"

# =============================================================================
# CONFIG — edit paths for your environment
# =============================================================================
SOURCE_ROOT   = "/data/VICTRE"                    # projections/, dicoms_v2/, manifest.csv
LOC_ROOT      = "/data/VICTRE/Locations/ext"       # extracted DIDSR/VICTRE Locations/*.tar.gz
FLATFIELD_DIR = "/data/VICTRE/flatfield"           # ff_base.npy, coefficients.json
OUTPUT_ROOT   = "/data/victre-paired"              # where train/val/test chunks are written
SPLIT_SOURCE  = None                                # optional: existing dataset to read the
                                                     # split assignment from. Precedence:
                                                     # splits.csv (released membership) >
                                                     # SPLIT_SOURCE > deterministic hash
LOCAL_TMP     = "/tmp/victre_paired_build"
os.makedirs(LOCAL_TMP, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True, choices=["train", "val", "test"])
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--n-shards", type=int, default=1)
ap.add_argument("--dry-run", type=int, default=0,
                help="process only this many patients, write to a scratch subfolder")
ap.add_argument("--force", action="store_true", help="reprocess already-complete chunks")
ap.add_argument("--selfcheck-every", type=int, default=25)
ap.add_argument("--diagnose", action="store_true",
                help="report the v9 scale and bounding-box checks per patient and write "
                     "diagnostics.json; implied by --dry-run")
args = ap.parse_args()
DIAGNOSE = bool(args.diagnose or args.dry_run)
_DIAG = []

OUT = os.path.join(OUTPUT_ROOT, "_dry_run") if args.dry_run else OUTPUT_ROOT
os.makedirs(os.path.join(OUT, args.split), exist_ok=True)

T0 = time.time()
def P(*a): print(" ".join(str(x) for x in a), flush=True)
def hdr(t): P("\n" + "=" * 78); P(f"{t}  [{(time.time()-T0)/60:.0f} min]"); P("=" * 78)

P(f"{'#'*78}\nVICTRE-Paired generation  split={args.split} shard={args.shard}/"
  f"{args.n_shards}{'  *** DRY RUN ***' if args.dry_run else ''}\n{'#'*78}")

# =============================================================================
# Flat-field (per-density polynomial correction + penumbra-strip threshold)
# =============================================================================
hdr("Flat-field")
coef_path = os.path.join(FLATFIELD_DIR, "coefficients.json")
if not os.path.exists(coef_path):
    sys.exit(f"Missing {coef_path}. See README, 'Flat-field'.")
COEF = json.load(open(coef_path))
# valley.json is no longer used: v9 does not remove a penumbra strip (see load_projections).
P("v9: no penumbra-strip threshold; valley.json is not read")

# native (pre-downsample) detector shape, from the VICTRE source geometry
NATIVE_DET_H, NATIVE_DET_W = 3000, 1500
_rr, _cc = np.meshgrid(np.arange(NATIVE_DET_H) / NATIVE_DET_H,
                       np.arange(NATIVE_DET_W) / NATIVE_DET_W, indexing="ij")
FF_BASE_PATH = os.path.join(FLATFIELD_DIR, "ff_base.npy")
if not os.path.exists(FF_BASE_PATH):
    sys.exit(f"Missing {FF_BASE_PATH} — the per-view base flat-field "
             f"(see README, 'Flat-field').")
FF_BASE = np.load(FF_BASE_PATH).astype(np.float32)   # (NA, 3000, 1500)

_ff_cache = {}
def build_flatfield(density):
    """Per-density polynomial correction applied to the base flat-field."""
    if density in _ff_cache:
        return _ff_cache[density]
    out = np.empty((NA, NATIVE_DET_H, NATIVE_DET_W), np.float32)
    for c in COEF[density]:
        view, b, deg = c["view"], np.array(c["coef"]), c["deg"]
        terms = [np.ones_like(_rr), _rr, _cc, _rr*_rr, _rr*_cc, _cc*_cc]
        if deg >= 3:
            terms += [_rr**3, _rr*_rr*_cc, _rr*_cc*_cc, _cc**3]
        out[view] = FF_BASE[view] * np.clip(sum(bi*ti for bi, ti in zip(b, terms)), 0.2, 3.0)
    _ff_cache[density] = out
    return out

# =============================================================================
# Source index: .loc files, manifest -> seed -> (loc path, is_positive, density, UID)
# =============================================================================
hdr("Indexing source data")
loc_files = glob.glob(f"{LOC_ROOT}/**/*.loc", recursive=True)
if not loc_files:
    sys.exit(f"No .loc files under {LOC_ROOT}. Clone https://github.com/DIDSR/VICTRE "
             f"and extract Locations/DBT-*.tar.gz there.")

def key_of(seed):
    return str(seed).lstrip("-")

seed2loc, seed2pos, seed2density = {}, {}, {}
for p in loc_files:
    m = re.match(r"roi_(SP|SA)_(-?\d+)\.loc", os.path.basename(p))
    if not m:
        continue
    k = key_of(m.group(2))
    seed2loc[k] = p
    seed2pos[k] = (m.group(1) == "SP")          # SP = signal-present, SA = signal-absent
    seed2density[k] = p.split("/DBT-")[1].split("/")[0]

manifest = pd.read_csv(os.path.join(SOURCE_ROOT, "manifest.csv"))
manifest["_k"] = manifest["PatientID"].astype(str).map(key_of)
recon_rows = manifest[manifest["SeriesDescription"].astype(str)
                      .str.contains("slice|DBT|recon", case=False, na=False)]
seed2uid = dict(zip(recon_rows["_k"], recon_rows["SeriesInstanceUID"].astype(str)))
P(f"  patients with .loc: {len(seed2loc)}  SP: {sum(seed2pos.values())}  "
  f"recon UID: {len(seed2uid)}")

# =============================================================================
# Split assignment
# =============================================================================
hdr("Split")
_split_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), SPLIT_CSV)
if os.path.exists(_split_csv):
    _df = pd.read_csv(_split_csv)
    seed2split = {int(s): str(sp) for s, sp in zip(_df["seed"], _df["split"])}
    P(f"  read from {SPLIT_CSV}: "
      f"{ {sp: sum(1 for v in seed2split.values() if v == sp) for sp in ['train','val','test']} }")
elif SPLIT_SOURCE:
    seed2split = {}
    for sp in ["train", "val", "test"]:
        for f in sorted(glob.glob(f"{SPLIT_SOURCE}/{sp}/*.npz")):
            d = np.load(f)
            for s in d["seed"]:
                seed2split[int(s)] = sp
            d.close()
    P(f"  read from {SPLIT_SOURCE}: "
      f"{ {sp: sum(1 for v in seed2split.values() if v == sp) for sp in ['train','val','test']} }")
else:
    # Fallback: deterministic 80/10/10 split by seed hash. Stable across re-runs,
    # but NOT the released membership -- use splits.csv for that.
    import hashlib
    def split_of(seed):
        h = int(hashlib.sha256(str(seed).encode()).hexdigest(), 16) % 100
        return "train" if h < 80 else ("val" if h < 90 else "test")
    seed2split = {int(k): split_of(k) for k in seed2loc}
    P(f"  derived: { {sp: sum(1 for v in seed2split.values() if v == sp) for sp in ['train','val','test']} }")

seeds_in_split = sorted(s for s, sp in seed2split.items() if sp == args.split)
seeds_mine = [s for i, s in enumerate(seeds_in_split) if i % args.n_shards == args.shard]
P(f"  {args.split}: {len(seeds_in_split)} total, {len(seeds_mine)} in this shard "
  f"({args.shard}/{args.n_shards})")
if args.dry_run:
    seeds_mine = seeds_mine[:args.dry_run]
    P(f"  dry-run: first {len(seeds_mine)} patients only, writing to {OUT}")

ITEMS = [(s, key_of(s), seed2uid[key_of(s)], seed2density[key_of(s)])
         for s in seeds_mine
         if key_of(s) in seed2loc and key_of(s) in seed2uid and key_of(s) in seed2density]
P(f"  valid (loc + uid + density present): {len(ITEMS)}/{len(seeds_mine)}")

# =============================================================================
# Core loaders
# =============================================================================
def downsample(x):
    return Fn.avg_pool2d(torch.from_numpy(x)[None, None].float(), DS)[0, 0].numpy()

def fit_to(a, h, w, pad=0.0):
    a = a[..., :min(a.shape[-2], h), :min(a.shape[-1], w)]
    hh, ww = a.shape[-2:]
    if hh < h or ww < w:
        a = np.pad(a, [(0, 0)]*(a.ndim - 2) + [(0, h - hh), (0, w - ww)], constant_values=pad)
    return a

def projection_files(seed):
    fs = sorted(glob.glob(f"{SOURCE_ROOT}/projections/{seed}/*/*.dcm"),
               key=lambda p: int(pydicom.dcmread(p, stop_before_pixels=True).InstanceNumber))
    return fs if len(fs) == NA else None

def load_projections(seed, density):
    """Real MC projections -> flat-field-corrected log-attenuation, NOT shifted.

    Mirrors the source pixel array through: illuminated-region bounding box,
    log(flatfield) - log(I) attenuation, 4x downsample, normalisation by the
    maximum. No empirical alignment shift is applied — the projection geometry
    needed to relate this to `clean` is written per patient instead (see
    geometry_for below / geom_* fields).

    v9 differs from v8 in two ways, both of them removals: there is no
    penumbra-strip threshold, and the scale is the maximum rather than the
    99.8th percentile, so no value is clipped.
    """
    files = projection_files(seed)
    if files is None:
        return None, None
    ff = build_flatfield(density)
    views = []
    diag = dict(tissue_outside_bbox=0.0, max_p=0.0, n_views=0)
    for view, f in enumerate(files):
        I = pydicom.dcmread(f).pixel_array.astype(np.float32)
        illuminated = I > AIR_THR
        ys, xs = np.where(illuminated)
        if len(ys) == 0:
            return None, None
        bbox = np.zeros_like(illuminated)
        bbox[ys.min():ys.max()+1, xs.min():xs.max()+1] = True
        p = np.clip(np.log(ff[view]) - np.log(np.clip(I, 1, None)), 0, None)
        # v9 CHANGE 1: no penumbra-strip removal. D1 showed the beam overfills the
        # detector (MC-GPU aperture 34.8 x 17.0 cm vs a 25.5 x 12.75 cm detector), so
        # there is no collimator shadow; D2 showed the readout floor is never reached
        # within the measured range (p stays usable to ~4.2 against thresholds at
        # 2.6-3.1). The threshold was deleting measurable tissue -- 17 % of it in dense
        # breasts. See overleaf/NUMBERS.md, "D1 + D2".
        if DIAGNOSE:
            # does the illuminated-region bounding box itself cut tissue away?
            tis = p > 0.10
            if tis.any():
                diag["tissue_outside_bbox"] += float((tis & ~bbox).sum()) / float(tis.sum())
            diag["max_p"] = max(diag["max_p"], float(p[bbox].max()))
            diag["n_views"] += 1
        p = np.where(bbox, p, 0.0)
        views.append(downsample(p))
        del I
    views = np.stack(views)
    if not (views > 0).any():
        return None, None
    # v9 CHANGE 2: normalise by the maximum, not the 99.8th percentile, so nothing
    # clips. D2 measured max/p99.8 = 1.013-1.075 once the strip is kept, so this costs
    # 1.3-7.5 % of the coding levels and removes the 0.2 % that used to saturate.
    # Rounded to float32 on the way out, and the noise is generated with this same
    # value. `proj_scale` is stored as float32; if generation used the float64 value
    # the user could not reproduce the stored noisy arrays from the record. The
    # difference is small but not zero -- measured at ~300-1900 pixels per array, 1 LSB
    # -- and it would make the bit-exactness claim false. Using the stored value makes
    # it true by construction.
    scale = float(np.float32(views.max() + 1e-6))
    views = fit_to(views / scale, PH, PW)
    if DIAGNOSE:
        diag["tissue_outside_bbox"] /= max(diag["n_views"], 1)
        _DIAG.append(dict(kind="proj", seed=seed, density=density, scale=scale,
                          max_p_native=diag["max_p"],
                          frac_tissue_outside_bbox=diag["tissue_outside_bbox"],
                          frac_at_ceiling=float((views >= 1.0).mean()),
                          pooled_shape=tuple(np.stack([downsample(np.zeros((NATIVE_DET_H, NATIVE_DET_W), np.float32))]).shape[1:]),
                          padded_to=(PH, PW)))
    return views.astype(np.float32), scale

def load_reconstruction(uid):
    """FBP reconstruction volume -> resampled to ZOUT slices, max-normalised (v9)."""
    files = list(glob.glob(f"{SOURCE_ROOT}/dicoms_v2/{uid}/*.dcm"))
    slices = sorted([(int(pydicom.dcmread(f, stop_before_pixels=True).InstanceNumber),
                      pydicom.dcmread(f).pixel_array.astype(np.float32)) for f in files],
                    key=lambda x: x[0])
    # VICTRE's own reconstructions are uint16 and some of them reach the ceiling. That is
    # the source's clipping, not ours, and it cannot be recovered after the fact because
    # v8's percentile normalisation removed the evidence. Recording it costs nothing and
    # turns an unknown into a number the paper can state.
    n_sat = sum(int((img >= 65535).sum()) for _, img in slices)
    n_tot = sum(int(img.size) for _, img in slices)
    src_sat = float(n_sat) / max(n_tot, 1)
    volume = np.stack([downsample(img) for _, img in slices])
    native_z = len(slices)
    t = torch.from_numpy(volume)[None, None]
    t = Fn.interpolate(t, size=(ZOUT, volume.shape[1], volume.shape[2]),
                       mode="trilinear", align_corners=False)[0, 0].numpy()
    # v9 CHANGE 3: normalise by the maximum, not the 99.5th percentile. D2 measured
    # max/p99.5 = 1.768 (worst 2.013) and D1 found 0.68 % of breast voxels sitting at
    # the ceiling -- the brightest structures were being flattened to roughly half
    # their value. This leaves about 37,000 levels for the old p99.5 range.
    # (recon_scale does not feed the noise model, but it is rounded the same way so the
    #  number used here and the number in the record are the same)
    scale = float(np.float32(t.max() + 1e-3))
    # The mask threshold is applied to the NORMALISED volume, so the normalisation
    # anchor decides its physical level. With MASK_ANCHOR = "p99.5" the threshold is
    # rescaled to sit exactly where the previous format put it, which makes the
    # breast support identical across formats; with "max" the literal 0.08 is used.
    # See constants.py for why this is a decision and not an implementation detail.
    mask_thr = MASK_THR
    if MASK_ANCHOR == "p99.5":
        mask_thr = MASK_THR * float(np.percentile(t, 99.5)) / max(scale, 1e-9)
    if DIAGNOSE:
        _DIAG.append(dict(kind="vol", uid=uid, scale=scale, native_z=native_z,
                          p995=float(np.percentile(t, 99.5)),
                          ratio_max_over_p995=float(t.max()/max(np.percentile(t, 99.5), 1e-9)),
                          frac_at_ceiling=float((t >= t.max()).mean()),
                          source_saturated=src_sat))
    return fit_to((t / scale).astype(np.float32), TH, TW), native_z, scale, src_sat, mask_thr

def load_lesions(key, native_z):
    """Lesion coordinates for signal-present (SP) patients only."""
    if not seed2pos.get(key, False):
        return np.zeros((0, 4), np.float32)
    path = seed2loc.get(key)
    if path is None:
        return np.zeros((0, 4), np.float32)
    loc = np.loadtxt(path)
    if loc.size == 0:
        return np.zeros((0, 4), np.float32)
    if loc.ndim == 1:
        loc = loc[None]
    rows = [[Z*ZOUT/native_z, X/DS, Y/DS, t] for X, Y, Z, t in loc
            if 0 <= Z*ZOUT/native_z < ZOUT and 0 <= X/DS < TH and 0 <= Y/DS < TW]
    return np.array(rows, np.float32) if rows else np.zeros((0, 4), np.float32)

def load_control_rois(key, native_z, max_rois=12):
    """Matched signal-absent control regions for SA (is_pos=False) patients only."""
    if seed2pos.get(key, True):
        return np.zeros((0, 5), np.float32)
    path = seed2loc.get(key)
    if path is None:
        return np.zeros((0, 5), np.float32)
    loc = np.loadtxt(path)
    if loc.size == 0:
        return np.zeros((0, 5), np.float32)
    if loc.ndim == 1:
        loc = loc[None]
    rows = []
    for X, Y, Z, roi_id in loc:
        z, h, w = Z*ZOUT/native_z, X/DS, Y/DS
        in_bounds = (0 <= z < ZOUT and 0 <= h < TH and 0 <= w < TW)
        rows.append([z, h, w, roi_id, float(in_bounds)])
    return np.array(rows[:max_rois], np.float32) if rows else np.zeros((0, 5), np.float32)

def to_uint16(a):
    return np.round(np.clip(a, 0, 1) * 65535).astype(np.uint16)

def geometry_for(density, native_z):
    """Per-patient LEAP volume placement — see constants.py for the derivation."""
    nx, ny = NATIVE_XY[density]
    vox_z = native_z / ZOUT
    offx = (TH * VOX_XY - nx * NATIVE_PIX) / 2.0 + OFFX_C
    offy = OFFY_A + OFFY_B * ny
    offz = -(SDD - SID) + (ZOUT * vox_z) / 2.0 + DELTA[density]
    return vox_z, offx, offy, offz

# =============================================================================
# Chunk writer (atomic: local tmp -> copy to destination tmp name -> rename)
# =============================================================================
def chunk_name(chunk_idx, split_name):
    # The shard is part of the name so that concurrent shards of the same split cannot
    # overwrite each other's chunks. Single-shard runs (val, test) use s0.
    return f"{split_name}_s{args.shard}_chunk_{chunk_idx:05d}.npz"

def write_chunk(buf, chunk_idx, split_name):
    fname = chunk_name(chunk_idx, split_name)
    local_path = os.path.join(LOCAL_TMP, fname)
    d = dict(
        clean=to_uint16(np.stack([b["clean"] for b in buf])),
        clean_proj=to_uint16(np.stack([b["proj"] for b in buf])),
        mask=np.stack([b["mask"] for b in buf]).astype(np.uint8),
        is_pos=np.array([b["is_pos"] for b in buf], bool),
        seed=np.array([b["seed"] for b in buf], np.int64),
        density=np.array([b["density"] for b in buf]),
        native_z=np.array([b["native_z"] for b in buf], np.int16),
        native_x=np.array([b["native_x"] for b in buf], np.int16),
        native_y=np.array([b["native_y"] for b in buf], np.int16),
        lesion_coords=np.stack([b["lesions"] for b in buf]),
        lesion_count=np.array([b["n_lesions"] for b in buf], np.int16),
        control_rois=np.stack([b["controls"] for b in buf]),
        control_count=np.array([b["n_controls"] for b in buf], np.int16),
        recon_scale=np.array([b["recon_scale"] for b in buf], np.float32),
        proj_scale=np.array([b["proj_scale"] for b in buf], np.float32),
        noise_seed=np.array([b["seed"] for b in buf], np.int64),
        geom_vox_z=np.array([b["vox_z"] for b in buf], np.float32),
        geom_offx=np.array([b["offx"] for b in buf], np.float32),
        geom_offy=np.array([b["offy"] for b in buf], np.float32),
        geom_offz=np.array([b["offz"] for b in buf], np.float32),
        # fraction of the SOURCE DICOM voxels sitting at the uint16 ceiling: VICTRE's own
        # clipping, recorded per patient so users can exclude or weight those cases
        recon_source_saturated=np.array([b["recon_src_sat"] for b in buf], np.float32),
        dose_levels=np.array(["full", "half", "quarter"]),
        dose_gains=np.array([DOSES["full"], DOSES["half"], DOSES["quarter"]], np.float32),
        elec_noise_photons=np.float32(ELEC_PHOTONS),
        noisy_range=np.array([NOISY_LO, NOISY_HI], np.float32),
        sid=np.float32(SID), sdd=np.float32(SDD), det_pix=np.float32(DET_PIX),
        format_version=np.array(FORMAT_VERSION),
        changes_from_v8=np.array(CHANGES_FROM_V8),
    )
    for tag in DOSES:
        # v9 CHANGE 4: the half-dose arrays carry their dose in the name like the other
        # two. v8 called them `noisy_proj` and `sigma`, which left the reader to infer
        # the budget from Table 1.
        proj_key  = f"noisy_proj_{tag}"
        sigma_key = f"sigma_{tag}"
        d[proj_key]  = np.stack([b["noisy"][tag] for b in buf])          # already uint16 (v8 encoding)
        d[sigma_key] = np.array([b["sigma"][tag] for b in buf], np.float32)

    np.savez_compressed(local_path, **d)
    final_path = os.path.join(OUT, split_name, fname)
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    tmp_dest = final_path + f".tmp{os.getpid()}"
    shutil.copy2(local_path, tmp_dest)
    os.replace(tmp_dest, final_path)          # atomic on the same filesystem
    os.remove(local_path)
    return final_path

N_KEYS_EXPECTED = 36      # fields in a v9 record == .npy members inside the .npz

def chunk_ok(path):
    """Does the chunk exist AND open?

    os.path.exists is not enough. The write path is local file -> copy to the
    output directory -> os.replace. The rename is atomic, but when the output
    directory is a network/FUSE mount the upload behind it is not: if the process
    or the mount dies mid-copy, a truncated .npz is left behind that raises
    BadZipFile when read. Treating that as complete would skip it forever and the
    run would finish with a hole in the data.

    The check is cheap: zipfile reads only the central directory at the end of the
    file and decompresses nothing. Truncation is exactly what breaks that.
    """
    if not os.path.exists(path):
        return False
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
    except Exception as e:
        P(f"  [corrupt] {os.path.basename(path)}: {type(e).__name__} -> will be rebuilt")
        return False
    if len(names) != N_KEYS_EXPECTED:
        P(f"  [incomplete] {os.path.basename(path)}: {len(names)}/{N_KEYS_EXPECTED} fields "
          f"-> will be rebuilt")
        return False
    return True


def chunk_exists(chunk_idx, split_name):
    return chunk_ok(os.path.join(OUT, split_name, chunk_name(chunk_idx, split_name)))

# =============================================================================
# Main loop
# =============================================================================
hdr(f"Generating {len(ITEMS)} patients")
split_name = args.split
# Resume. Chunk i always holds ITEMS[CHUNK*i : CHUNK*(i+1)], so resuming means
# skipping the chunks already on disk up to the first gap and cutting the list
# there. Doing the skip inside the per-patient loop -- as this did until it was
# caught in production -- consumes one ITEM per skipped CHUNK, so every patient in
# ITEMS[K : CHUNK*K] is written a second time into a new chunk and the shard ends
# up with far more chunks than patients.
start_chunk = 0
if not args.force:
    while chunk_exists(start_chunk, split_name):
        start_chunk += 1
todo = ITEMS[CHUNK * start_chunk:]
buf, chunk_idx, skipped, issues = [], start_chunk, 0, []
done, n_new = CHUNK * start_chunk, 0        # done: absolute · n_new: this session
if start_chunk:
    P(f"  [resume] {start_chunk} complete chunks -> first {CHUNK*start_chunk} patients "
      f"skipped, {len(todo)} to go")
t0 = time.time()

for idx, (seed, key, uid, density) in enumerate(todo):
    try:
        proj, proj_scale = load_projections(seed, density)
        if proj is None:
            P(f"  ! {seed}: could not read projections / no air region, skipping")
            skipped += 1
            continue
        clean, native_z, recon_scale, recon_src_sat, mask_thr = load_reconstruction(uid)
        mask = (clean > mask_thr)

        lesions_raw = load_lesions(key, native_z)
        lesions = np.zeros((8, 4), np.float32)
        n_lesions = min(len(lesions_raw), 8)
        if n_lesions:
            lesions[:n_lesions] = lesions_raw[:n_lesions]

        controls_raw = load_control_rois(key, native_z)
        controls = np.zeros((12, 5), np.float32)
        n_controls = min(len(controls_raw), 12)
        if n_controls:
            controls[:n_controls] = controls_raw[:n_controls]

        # noise is generated from the quantised clean_proj (what the record stores),
        # so the released noisy arrays regenerate bit-for-bit from the record
        proj_u16 = to_uint16(proj); proj_q = proj_u16.astype(np.float32) / 65535.0
        noisy, sigma = {}, {}
        for tag, gain in DOSES.items():
            n_u16 = encode_noisy(add_noise(proj_u16, proj_scale, gain, seed, DOSE_IDX[tag]))
            noisy[tag] = n_u16
            support = proj_q > 0.02
            sigma[tag] = float((decode_noisy(n_u16) - proj_q)[support].std()) if support.any() else 0.0

        vox_z, offx, offy, offz = geometry_for(density, native_z)
        native_x, native_y = NATIVE_XY[density]

        if args.selfcheck_every > 0 and idx % args.selfcheck_every == 0:
            problems = []
            if not np.all(np.isfinite(clean)): problems.append("clean_nan")
            if not np.all(np.isfinite(proj)):  problems.append("clean_proj_nan")
            if mask.sum() < 50: problems.append(f"mask_almost_empty({int(mask.sum())})")
            if n_lesions > 0 and not np.all(lesions[n_lesions:] == 0):
                problems.append("lesion_padding_wrong")
            if n_controls > 0 and not np.all(controls[n_controls:] == 0):
                problems.append("control_padding_wrong")
            if problems:
                issues.append(dict(seed=seed, density=density, problems=problems))
                P(f"  ! self-check: seed={seed} -> {problems}")

        buf.append(dict(clean=clean, proj=proj, mask=mask,
                        is_pos=seed2pos.get(key, False), seed=int(seed), density=density,
                        native_z=native_z, native_x=native_x, native_y=native_y,
                        lesions=lesions, n_lesions=n_lesions,
                        controls=controls, n_controls=n_controls,
                        recon_scale=recon_scale, proj_scale=proj_scale,
                        recon_src_sat=recon_src_sat,
                        noisy=noisy, sigma=sigma,
                        vox_z=vox_z, offx=offx, offy=offy, offz=offz))
        done += 1
        n_new += 1
        elapsed = time.time() - t0
        rate = elapsed / max(1, n_new)      # this session only; `done` counts skipped chunks too
        remaining = max(0, len(ITEMS) - done - skipped) * rate
        P(f"  [{done}/{len(ITEMS)}] seed={seed} {density:>12s} done | "
          f"{elapsed/3600:.2f} h elapsed | ~{remaining/3600:.1f} h remaining | "
          f"{rate:.0f} s/patient")
    except Exception as e:
        P(f"  ! ERROR seed={seed}: {type(e).__name__}: {e}")
        issues.append(dict(seed=seed, density=density, problems=[f"exception:{e}"]))
        skipped += 1
        continue

    if len(buf) >= CHUNK:
        path = write_chunk(buf, chunk_idx, split_name)
        P(f"  >>> chunk {chunk_idx} written -> {path}")
        buf = []
        chunk_idx += 1
    gc.collect()

if DIAGNOSE and _DIAG:
    dp = os.path.join(OUT, f"diagnostics_{args.split}.json")
    json.dump(_DIAG, open(dp, "w"), indent=1, default=float)
    pr = [d for d in _DIAG if d["kind"] == "proj"]
    vo = [d for d in _DIAG if d["kind"] == "vol"]
    hdr("v9 diagnostics")
    if pr:
        P(f"  projections, n={len(pr)}")
        P(f"    tissue outside the illuminated bounding box : "
          f"{100*max(d['frac_tissue_outside_bbox'] for d in pr):.4f} % (worst patient)  "
          f"<- must be ~0, otherwise the box itself is cutting tissue")
        P(f"    highest attenuation kept (native)           : "
          f"{max(d['max_p_native'] for d in pr):.2f}  (v8 cut everything above 2.6-3.1)")
        P(f"    pixels at the coding ceiling                : "
          f"{100*max(d['frac_at_ceiling'] for d in pr):.4f} %  <- one pixel per patient by construction")
        P(f"    pooled {pr[0]['pooled_shape']} padded to {pr[0]['padded_to']} (padding goes on the bottom/right)")
        P(f"    proj_scale range                            : "
          f"{min(d['scale'] for d in pr):.3f} .. {max(d['scale'] for d in pr):.3f}")
    if vo:
        P(f"  volumes, n={len(vo)}")
        P(f"    max / p99.5                                 : "
          f"{np.mean([d['ratio_max_over_p995'] for d in vo]):.3f}  "
          f"(worst {max(d['ratio_max_over_p995'] for d in vo):.3f})")
        P(f"    recon_scale range                           : "
          f"{min(d['scale'] for d in vo):.3f} .. {max(d['scale'] for d in vo):.3f}")
    P(f"  written to {dp}")

if buf:
    path = write_chunk(buf, chunk_idx, split_name)
    P(f"  [final chunk {chunk_idx}] -> {path}  ({len(buf)} patients, partial)")
    chunk_idx += 1

hdr("Summary")
P(f"  split: {split_name}{'  (dry run)' if args.dry_run else ''}")
P(f"  processed: {done}  skipped: {skipped}  chunks written: {chunk_idx}")
P(f"  total time: {(time.time()-T0)/3600:.2f} h")
if issues:
    P(f"\n  {len(issues)} patients flagged by self-check (data was written; review):")
    for x in issues[:30]:
        P(f"    seed={x['seed']} {x['density']}: {x['problems']}")
    if len(issues) > 30:
        P(f"    ... and {len(issues)-30} more")
else:
    P("\n  Self-check found no issues.")

summary_path = os.path.join(OUT, f"generation_summary_{split_name}_shard{args.shard}.json")
json.dump(dict(split=split_name, shard=args.shard, n_shards=args.n_shards,
               processed=done, skipped=skipped, chunks=chunk_idx, issues=issues,
               dry_run=bool(args.dry_run), hours=(time.time()-T0)/3600),
          open(summary_path, "w"), indent=2, default=str)
P(f"\n-> {summary_path}")
if args.dry_run:
    P(f"\n  This was a dry run. Nothing was written to {OUTPUT_ROOT}.")
    P(f"  Inspect output under {OUT}/{split_name}/, then re-run without --dry-run.")
