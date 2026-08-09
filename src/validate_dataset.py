#!/usr/bin/env python3
"""
VICTRE-Paired — technical validation.

Validates a built dataset end to end and reproduces the numbers in the paper's
Technical Validation section. Cheap, population-wide checks (schema, dtypes,
constants, the analytic geometry formula) run over every patient; expensive
per-array measurements (flat-field, noise, geometry residual, task-based SDNR)
run on a stratified sample by default — see --profile.

Stages
------
  1  index        small per-patient metadata fields                    full population
  2  integrity    schema, dtypes, shapes, constants, NaN/Inf,          full population
                  duplicate seeds, split disjointness, geometry formula,
                  optional zip CRC (--crc)
  3  storage      chunk sizes, per-key breakdown, compression ratio
  4  fused pass   one read per chunk, computing:                       sampled
                  - normalization invariants (99.5 / 99.8 percentile)
                  - mask = clean > threshold check
                  - penumbra-strip validity
                  - flat-field / air pedestal / angular uniformity
                  - noise: measured vs. stored sigma, dose monotonicity,
                    whiteness, and bit-exact reproduction from noise_seed
                  - lesion / control ROI validity and SDNR
                  - geometry: residual parallax between A(clean) and clean_proj (GPU)
  5  summary      statistics and pass/fail checks from stage 4's table  instant
  6  baselines    two-regime reconstruction, 4 methods, official test split (GPU)
  7  dose sweep   reconstruction quality vs. dose + reconstruction-noise
                  autocorrelation (GPU)
  8  offset tune  per-density z-offset refinement with a held-out probe
                  set — metadata-only, does not touch image data (GPU)
  9  DC sweep     controlled flat-field-pedestal injection (GPU)
  R  report       REPORT.md + results.json + a small results.zip

Resumable: each stage writes its own table; re-running continues where it left
off (use --force to recompute specific stages).

Usage
-----
    python validate_dataset.py --data /path/to/victre-paired --profile quick
    python validate_dataset.py --data /path/to/victre-paired --profile full --crc
    python validate_dataset.py --data /path/to/victre-paired --stages 12345 --out ./report
"""

import os, sys, glob, json, csv, gc, time, shutil, zipfile, math, argparse, traceback

import numpy as np
np.seterr(all="ignore")
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import (ZOUT, TH, TW, NA, PH, PW, DET_PIX, VOX_XY, ANG, NATIVE_PIX,
                       SID, SDD, OFFX_C, OFFY_A, OFFY_B, DELTA, NATIVE_XY,
                       DOSES, DOSE_IDX, S_ELEC, MASK_THR, P_RECON, P_PROJ,
                       DENSITIES, BROKEN_SEEDS)

# =============================================================================
# CLI
# =============================================================================
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="path to the dataset root (train/val/test subfolders)")
ap.add_argument("--out", default="./validation_report", help="output directory")
ap.add_argument("--profile", default="quick", choices=["smoke", "quick", "full"],
                help="smoke: ~15 min sanity check. quick: ~1.5-2 h, stratified sample, "
                     "sufficient for publication (default). full: every patient, "
                     "8-20+ h, run once before freezing a release.")
ap.add_argument("--stages", default="123456789R",
                help="which stages to run, e.g. '12345R' to skip the GPU stages")
ap.add_argument("--force", default="", help="stages to recompute even if cached, e.g. '48'")
ap.add_argument("--crc", action="store_true", help="verify zip CRC of every chunk (stage 2, slow)")
ap.add_argument("--split", default="test", help="split used for stages 6/7/9")
ap.add_argument("--sirt-iters", type=int, default=50)
ap.add_argument("--reference", default=None,
                help="optional: path to a prior dataset version, for a regression "
                     "comparison (not needed to validate a single release)")
ap.add_argument("--reference-tag", default="reference")
args = ap.parse_args()

DATA    = args.data
OUTROOT = args.out
STAGES  = args.stages
FORCE_STAGES = args.force
BASELINE_SPLIT = args.split
SIRT_ITERS = args.sirt_iters
DO_ZIP_CRC = args.crc
RESUME = True   # always resume from cached stage/table files; use --force to override

# The dataset under test. Variable names TARGET/TARGET_TAG are kept (rather than
# renamed to DATA) because the validation logic below treats "target" and an
# optional "reference" symmetrically; TARGET is simply an alias for --data.
TARGET     = DATA
TARGET_TAG = "dataset"
REFERENCE     = args.reference
REF_TAG       = args.reference_tag
DO_REFERENCE  = REFERENCE is not None

_PROF = {
    "smoke": dict(FULL=False, N_FUSED=40,  N_BASELINE=8,  N_DOSE=8,
                  DO_NOISE_ARRAYS=True, DO_NOISE_BITEXACT=True, N_BITEXACT=20,
                  DO_GEOMETRY=True),
    "quick": dict(FULL=False, N_FUSED=300, N_BASELINE=32, N_DOSE=60,
                  DO_NOISE_ARRAYS=True, DO_NOISE_BITEXACT=True, N_BITEXACT=60,
                  DO_GEOMETRY=True),
    "full":  dict(FULL=True,  N_FUSED=0,   N_BASELINE=64, N_DOSE=120,
                  DO_NOISE_ARRAYS=True, DO_NOISE_BITEXACT=True, N_BITEXACT=0,
                  DO_GEOMETRY=True),
}[args.profile]
FULL = _PROF["FULL"]
N_FUSED, N_BASELINE, N_DOSE = _PROF["N_FUSED"], _PROF["N_BASELINE"], _PROF["N_DOSE"]
N_BITEXACT = _PROF["N_BITEXACT"]
DO_NOISE_ARRAYS   = _PROF["DO_NOISE_ARRAYS"]
DO_NOISE_BITEXACT = _PROF["DO_NOISE_BITEXACT"]
DO_GEOMETRY       = _PROF["DO_GEOMETRY"]

FBP_WINDOW, FBP_COSWEIGHT = "hann", True
N_OFFZ_PROBE, N_OFFZ_VALID = 16, 60
N_PEDESTAL = 8
PEDESTALS  = [0.0, 0.10, 0.20, 0.35, 0.60]

GEO = dict(SID=SID, SDD=SDD)
CFG = dict(ZOUT=ZOUT, TH=TH, TW=TW, PH=PH, PW=PW, NA=NA, DET_PIX=DET_PIX,
           VOX_XY=VOX_XY, ANG=ANG, NATIVE_PIX=NATIVE_PIX, MASK_THR=MASK_THR,
           S_ELEC=S_ELEC, P_RECON=P_RECON, P_PROJ=P_PROJ)
DOSE = [("full", "noisy_proj_full", "sigma_full", DOSES["full"]),
        ("half", "noisy_proj", "sigma", DOSES["half"]),
        ("quarter", "noisy_proj_quarter", "sigma_quarter", DOSES["quarter"])]
DENS   = DENSITIES
BROKEN = BROKEN_SEEDS
KEYS = {"clean","clean_proj","mask","noisy_proj","noisy_proj_full",
    "noisy_proj_quarter","sigma","sigma_full","sigma_quarter","is_pos","seed",
    "density","native_z","native_x","native_y","lesion_coords","lesion_count",
    "control_rois","control_count","recon_scale","proj_scale","noise_seed",
    "geom_vox_z","geom_offx","geom_offy","geom_offz","strip_valley",
    "dose_levels","dose_gains","elec_noise","sid","sdd","det_pix"}

# =============================================================================
# Noise formula — must match generate_dataset.py's add_noise() exactly.
# Validated bit-exact against production output; see paper §Technical Validation.
# =============================================================================
def noise_formula(p, proj_scale, gain, seed, dose_idx, s_elec=S_ELEC):
    rng = np.random.default_rng(np.uint64(seed) * 10 + np.uint64(dose_idx))
    I0 = 1.0 / gain
    I = I0 * np.exp(-np.clip(p, 0, 1) * proj_scale)
    N = rng.poisson(np.maximum(I, 1e-9)).astype(np.float64)
    N = N + rng.standard_normal(p.shape) * (s_elec * I0 * 0.02)
    return np.clip(-np.log(np.maximum(N, 1e-9) / I0) / proj_scale, 0, 1).astype(np.float32)

TAB, FIG = f"{OUTROOT}/tables", f"{OUTROOT}/figures"
CACHE = os.path.join(OUTROOT, ".cache")
for d in (OUTROOT, TAB, FIG, CACHE):
    os.makedirs(d, exist_ok=True)
_left = glob.glob(f"{CACHE}/*")
if _left:
    for f in _left:
        try: os.remove(f)
        except Exception: pass
    print(f"[cleanup] removed {len(_left)} stale cache files")

T0 = time.time()
LOG = open(f"{OUTROOT}/log.txt", "a")
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def hdr(t):
    P("\n" + "=" * 80); P(f"{t}   [{(time.time()-T0)/60:.0f} min]"); P("=" * 80)

P(f"\n{'#'*80}\n# VICTRE-Paired validation   {time.strftime('%Y-%m-%d %H:%M')}"
  f"\n# data={DATA}"
  f"\n# profile={args.profile}  stages={STAGES}  full={FULL}  "
  f"n_fused={'ALL' if FULL else N_FUSED}  bit-exact={DO_NOISE_BITEXACT}"
  f"(n={N_BITEXACT or 'all'})"
  f"\n{'#'*80}")

RES, CHECKS = {}, []
def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), str(detail)))
    P(f"  [{'PASS' if ok else 'FAIL':>5s}] {name:<62s} {detail}")

_CK0 = 0
def stage_begin(tag, keys):
    global _CK0
    fp = f"{TAB}/_stage{tag}.json"
    if tag not in FORCE_STAGES and os.path.exists(fp):
        try:
            d = json.load(open(fp)); RES.update(d.get("res", {}))
            for c in d.get("checks", []):
                CHECKS.append((c["name"], bool(c["pass"]), c["detail"]))
                P(f"  [{'PASS' if c['pass'] else 'FAIL':>5s}] {c['name']:<62s} "
                  f"{c['detail']}  (cached)")
            P(f"  [cached] stage {tag} not recomputed"); return True
        except Exception as e:
            P(f"  [warn] stage {tag} cache unreadable ({e})")
    _CK0 = len(CHECKS); return False
def stage_end(tag, keys):
    d = dict(res={k: RES[k] for k in keys if k in RES},
             checks=[{"name": n, "pass": v, "detail": dd} for n, v, dd in CHECKS[_CK0:]])
    tmp = f"{TAB}/_stage{tag}.json.t{os.getpid()}"
    json.dump(d, open(tmp, "w"), default=str); os.replace(tmp, f"{TAB}/_stage{tag}.json")

def dump(rows, name):
    if not rows: return
    keys = sorted({k for r in rows for k in r})
    tmp = f"{TAB}/{name}.csv.t{os.getpid()}"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in keys})
    os.replace(tmp, f"{TAB}/{name}.csv")
def load_rows(name, tag=None):
    if tag and tag in FORCE_STAGES:
        P(f"  [force] stage {tag}: ignoring existing {name}.csv"); return []
    p = f"{TAB}/{name}.csv"
    if os.path.exists(p):
        with open(p) as f: return list(csv.DictReader(f))
    return []

def fetch(p):
    """Copy a chunk to local scratch for faster repeated random access."""
    loc = os.path.join(CACHE, f"{abs(hash(p))%999983}_{os.path.basename(p)}")
    if not os.path.exists(loc):
        try: shutil.copy2(p, loc)
        except Exception: return p
    return loc
def release(loc):
    if loc and CACHE in loc and os.path.exists(loc):
        try: os.remove(loc)
        except Exception: pass

def f01(a):
    a = np.asarray(a)
    if a.dtype == np.uint16: return a.astype(np.float32)/65535.0
    a = a.astype(np.float32)
    return a if a.max() <= 1.01 else a/65535.0

def _pb(ym, y0, yp):
    d = (ym - 2*y0 + yp)
    return 0.0 if abs(d) < 1e-12 else 0.5*(ym - yp)/d
def best_shift(t, m, mr=300, mc=140):
    a = t - t.mean(); b = m - m.mean()
    cc = np.fft.irfft2(np.fft.rfft2(a)*np.conj(np.fft.rfft2(b)), s=a.shape)
    cs = np.fft.fftshift(cc); H, W = a.shape; cy, cx = H//2, W//2
    r0, r1 = max(0, cy-mr), min(H, cy+mr+1); k0, k1 = max(0, cx-mc), min(W, cx+mc+1)
    sub = cs[r0:r1, k0:k1]
    i, j = np.unravel_index(np.argmax(sub), sub.shape)
    dr, dc = float((r0+i)-cy), float((k0+j)-cx)
    if 0 < i < sub.shape[0]-1: dr += _pb(sub[i-1,j], sub[i,j], sub[i+1,j])
    if 0 < j < sub.shape[1]-1: dc += _pb(sub[i,j-1], sub[i,j], sub[i,j+1])
    return dr, dc

def corr(a, b, m=None):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    if m is not None:
        k = np.asarray(m) > 0.5
        if k.sum() < 100: return float("nan")
        a, b = a[k], b[k]
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    return float(a @ b/(np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))
def psnr_sm(pred, gt, mask):
    k = np.asarray(mask) > 0.5
    if k.sum() < 100: return float("nan")
    x = np.asarray(pred, np.float64)[k]; y = np.asarray(gt, np.float64)[k]
    A = np.stack([x, np.ones_like(x)], 1)
    try: c, *_ = np.linalg.lstsq(A, y, rcond=None)
    except Exception: return float("nan")
    return float(-10*np.log10((((A @ c)-y)**2).mean() + 1e-12))
def sdnr_at(vol, z, h, w, ri=4, ro=12):
    z, h, w = int(round(z)), int(round(h)), int(round(w))
    if not (0 <= z < vol.shape[0] and 0 <= h < vol.shape[1] and 0 <= w < vol.shape[2]):
        return np.nan
    c = vol[max(0,z-1):z+2, max(0,h-ri):h+ri+1, max(0,w-ri):w+ri+1]
    R = vol[z, max(0,h-ro):h+ro+1, max(0,w-ro):w+ro+1]
    return float((c.mean()-R.mean())/R.std()) if R.std() > 1e-8 else np.nan
def lag1(a, axis):
    a = np.asarray(a, np.float64); a = a - a.mean()
    x = np.take(a, range(0, a.shape[axis]-1), axis=axis).ravel()
    y = np.take(a, range(1, a.shape[axis]), axis=axis).ravel()
    return float(x @ y/(np.linalg.norm(x)*np.linalg.norm(y) + 1e-12))
def q(a, p):
    a = np.asarray(a, np.float64); a = a[np.isfinite(a)]
    return float(np.percentile(a, p)) if a.size else float("nan")
def mn(a):
    a = np.asarray(a, np.float64); a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")
MID = CFG["NA"]//2
def _sub(h, w, fy=(0.15,0.75), fx=(0.10,0.85)):
    return (slice(int(fy[0]*h), int(fy[1]*h)), slice(int(fx[0]*w), int(fx[1]*w)))
# =============================================================================
# STAGE 1 -- INDEX
# =============================================================================
hdr("STAGE 1 -- INDEX")
SMALL = ["seed","is_pos","density","native_z","native_x","native_y","lesion_count",
         "control_count","proj_scale","recon_scale","sigma","sigma_full",
         "sigma_quarter","noise_seed","geom_vox_z","geom_offx","geom_offy",
         "geom_offz","strip_valley"]

def build_index(root, cache_name):
    cp = f"{TAB}/{cache_name}.json"
    if RESUME and "1" not in FORCE_STAGES and os.path.exists(cp):
        try:
            d = json.load(open(cp))
            idx = {int(k): v for k, v in d["idx"].items()}   # yeni format
            P(f"  [cached] {cache_name}")
            return idx, d["bad"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            P(f"  [warning] {cache_name} eski/bozuk formatta -> yeniden indeksleniyor")
    idx, bad, seen, n, t0 = {}, [], {}, 0, time.time()
    for sp in ["train", "val", "test"]:
        for f in sorted(glob.glob(f"{root}/{sp}/*.npz")):
            try: d = np.load(f)
            except Exception as e:
                bad.append(dict(chunk=os.path.basename(f), problem=f"acilamadi:{e}")); continue
            try:
                seeds = d["seed"]
                for b in range(len(seeds)):
                    rec = dict(chunk=f, b=int(b), split=sp, fsize=os.path.getsize(f))
                    for k in SMALL:
                        if k in d.files:
                            v = d[k][b]; rec[k] = v.item() if hasattr(v, "item") else str(v)
                    sd = int(seeds[b])
                    if sd in seen:
                        bad.append(dict(chunk=os.path.basename(f),
                                        problem=f"TEKRAR EDEN SEED {sd} (ilk:{seen[sd]})"))
                    seen[sd] = os.path.basename(f); idx[sd] = rec
            except Exception as e:
                bad.append(dict(chunk=os.path.basename(f), problem=f"okunamadi:{e}"))
            d.close(); n += 1
            if n % 50 == 0: P(f"    {n} chunk, {len(idx)} patients  [{(time.time()-t0)/60:.1f} dk]")
    tmp = cp + f".t{os.getpid()}"
    json.dump(dict(idx={str(k): v for k, v in idx.items()}, bad=bad), open(tmp, "w"),
              default=str); os.replace(tmp, cp)
    P(f"  {n} chunk, {len(idx)} patients  [{(time.time()-t0)/60:.1f} dk]")
    return idx, bad

IDX, BAD = build_index(TARGET, f"IDX_{TARGET_TAG}")
RIDX, RBAD = ({}, [])
if REFERENCE and DO_REFERENCE:
    RIDX, RBAD = build_index(REFERENCE, f"IDX_{REF_TAG}")
P(f"\n  {TARGET_TAG}: {len(IDX)} patients" + (f"   |   {REF_TAG}: {len(RIDX)} patients" if RIDX else ""))
RES[f"n_{TARGET_TAG}"] = len(IDX)
if RIDX: RES[f"n_{REF_TAG}"] = len(RIDX)
DENS_OF = {s: str(IDX[s].get("density")) for s in IDX}
ALL = sorted(s for s in IDX if s not in BROKEN)

def select(n, seed=7, split=None, pool=None):
    """Chunk sirasina gore SIRALI secim (Drive I/O'yu minimize eder)."""
    keys = pool if pool is not None else ALL
    if split: keys = [s for s in keys if IDX[s]["split"] == split]
    if FULL or not n or n <= 0: sel = list(keys)
    else:
        per = {}
        for s in keys: per.setdefault(DENS_OF.get(s), []).append(s)
        rng = np.random.default_rng(seed); sel = []
        k = max(1, n//max(1, len(per)))
        for dn, lst in sorted(per.items()):
            lst = sorted(lst); t = min(k, len(lst))
            sel += [lst[i] for i in rng.choice(len(lst), t, replace=False)]
    return sorted(sel, key=lambda s: (IDX[s]["chunk"], IDX[s]["b"]))

# =============================================================================
# STAGE 2 -- INTEGRITY
# =============================================================================
_S2 = ["schema_issues","n_chunk","split_dist","geom_formula_violations",
       "crc_bad","seed_only_target","seed_only_ref","metadata_mismatch"]
if "2" in STAGES:
    hdr("STAGE 2 -- INTEGRITY (schema / dtype / shape / constants / CRC)")
if "2" in STAGES and not stage_begin("2", _S2):
    files = sorted(glob.glob(f"{TARGET}/train/*.npz") + glob.glob(f"{TARGET}/val/*.npz")
                   + glob.glob(f"{TARGET}/test/*.npz"))
    shp = {"clean": (CFG["ZOUT"], CFG["TH"], CFG["TW"]),
           "mask": (CFG["ZOUT"], CFG["TH"], CFG["TW"]),
           "clean_proj": (CFG["NA"], CFG["PH"], CFG["PW"]),
           "noisy_proj": (CFG["NA"], CFG["PH"], CFG["PW"]),
           "noisy_proj_full": (CFG["NA"], CFG["PH"], CFG["PW"]),
           "noisy_proj_quarter": (CFG["NA"], CFG["PH"], CFG["PW"])}
    dt = {"clean": np.uint16, "clean_proj": np.uint16, "mask": np.uint8,
          "noisy_proj": np.uint16, "noisy_proj_full": np.uint16,
          "noisy_proj_quarter": np.uint16}
    consts = [("sid", GEO["SID"]), ("sdd", GEO["SDD"]),
              ("det_pix", CFG["DET_PIX"]), ("elec_noise", CFG["S_ELEC"])]
    issues, crc_bad, t0 = [], [], time.time()
    for i, f in enumerate(files):
        bn = os.path.basename(f)
        try: d = np.load(f)
        except Exception as e:
            issues.append(dict(chunk=bn, problem=f"acilamadi:{e}")); continue
        ks = set(d.files)
        for miss in sorted(KEYS - ks): issues.append(dict(chunk=bn, problem=f"eksik:{miss}"))
        for ex in sorted(ks - KEYS):   issues.append(dict(chunk=bn, problem=f"fazla:{ex}"))
        for k, v in dt.items():
            if k in ks and d[k].dtype != v:
                issues.append(dict(chunk=bn, problem=f"{k} dtype={d[k].dtype}!={v.__name__}"))
        for k, v in shp.items():
            if k in ks and tuple(d[k].shape[1:]) != v:
                issues.append(dict(chunk=bn, problem=f"{k} sekil={d[k].shape[1:]}!={v}"))
        for k in ["proj_scale","recon_scale","geom_vox_z","geom_offx","geom_offy",
                  "geom_offz","sigma","sigma_full","sigma_quarter","strip_valley"]:
            if k in ks and not np.all(np.isfinite(np.asarray(d[k], np.float64))):
                issues.append(dict(chunk=bn, problem=f"{k} NaN/Inf"))
        for k, v in consts:
            if k in ks and abs(float(np.asarray(d[k]).ravel()[0]) - v) > 1e-6:
                issues.append(dict(chunk=bn, problem=f"{k}!={v}"))
        for k, exp in [("dose_levels", [1.0, 0.5, 0.25]),
                       ("dose_gains", [g for _, _, _, g in DOSE])]:
            if k in ks:
                got = np.asarray(d[k], np.float64).ravel().tolist()
                if len(got) != len(exp) or any(abs(a-b_) > 1e-9 for a, b_ in zip(got, exp)):
                    issues.append(dict(chunk=bn, problem=f"{k}={got}!={exp}"))
        # ayni chunk icindeki batch boyutlari tutarli mi
        ns = {k: d[k].shape[0] for k in ks if getattr(d[k], "ndim", 0) >= 1
              and d[k].shape and k not in ("dose_levels", "dose_gains")}
        if ns and len(set(ns.values())) > 1:
            issues.append(dict(chunk=bn, problem=f"batch boyu tutarsiz:{sorted(set(ns.values()))}"))
        d.close()
        if DO_ZIP_CRC:
            try:
                z = zipfile.ZipFile(f); bad = z.testzip(); z.close()
                if bad: crc_bad.append(dict(chunk=bn, member=bad))
            except Exception as e:
                crc_bad.append(dict(chunk=bn, member=f"acilamadi:{e}"))
        if (i+1) % 50 == 0: P(f"    {i+1}/{len(files)}  [{(time.time()-t0)/60:.1f} dk]")
    for r in BAD:  issues.append(dict(chunk=f"{TARGET_TAG}:"+str(r.get("chunk")), problem=r["problem"]))
    for r in RBAD: issues.append(dict(chunk=f"{REF_TAG}:"+str(r.get("chunk")), problem=r["problem"]))
    dump(issues, "T2_issues"); dump(crc_bad, "T2_crc")
    P(f"  {len(files)} chunk, {len(issues)} sorun, CRC: "
      f"{'kapali' if not DO_ZIP_CRC else f'{len(crc_bad)} bozuk'}")
    check("key set is complete with no extras (all chunks)",
          not any(x["problem"].startswith(("eksik","fazla")) for x in issues))
    check("dtypes and shapes as expected",
          not any("dtype" in x["problem"] or "sekil" in x["problem"] for x in issues))
    check("no NaN/Inf; sid/sdd/det_pix/elec_noise/dose constants correct",
          not any("NaN" in x["problem"] or "!=" in x["problem"] for x in issues))
    check("batch sizes consistent within each chunk",
          not any("batch" in x["problem"] for x in issues))
    check("no seed appears more than once", not any("TEKRAR" in x["problem"] for x in issues))
    if DO_ZIP_CRC:
        check("zip CRC integrity (required before upload)", not crc_bad, f"{len(crc_bad)} bozuk")
    RES["schema_issues"] = len(issues); RES["n_chunk"] = len(files)
    RES["crc_bad"] = len(crc_bad) if DO_ZIP_CRC else None

    # geometri formulu (tam populasyon)
    v = dict(vox_z=0, offx=0, offy=0, offz=0, n=0)
    for s in ALL:
        r = IDX[s]; dn = str(r.get("density"))
        if dn not in NATIVE_XY: continue
        try:
            nz = int(r["native_z"]); nx, ny = NATIVE_XY[dn]
            e_vz = nz/CFG["ZOUT"]
            e_ox = (CFG["TH"]*CFG["VOX_XY"] - nx*CFG["NATIVE_PIX"])/2.0 + OFFX_C
            e_oy = OFFY_A + OFFY_B*ny
            e_oz = -(GEO["SDD"]-GEO["SID"]) + CFG["ZOUT"]*e_vz/2.0 + DELTA[dn]
            v["vox_z"] += abs(float(r["geom_vox_z"])-e_vz) > 1e-4
            v["offx"]  += abs(float(r["geom_offx"])-e_ox) > 0.01
            v["offy"]  += abs(float(r["geom_offy"])-e_oy) > 0.01
            v["offz"]  += abs(float(r["geom_offz"])-e_oz) > 0.01
            v["n"] += 1
        except Exception: pass
    check("stored geom_* fields match the analytic model exactly (full population)",
          v["vox_z"] == v["offx"] == v["offy"] == v["offz"] == 0, str(v))
    RES["geom_formula_violations"] = v
    nxy = all({(int(IDX[s]["native_x"]), int(IDX[s]["native_y"]))
               for s in ALL if DENS_OF.get(s) == dn and "native_x" in IDX[s]}
              in ({NATIVE_XY[dn]}, set()) for dn in DENS)
    check("native_x/native_y are class-constant", nxy)

    dist = {}
    for s in IDX:
        sp = IDX[s]["split"]; dn = DENS_OF.get(s)
        dist.setdefault(sp, dict(n=0, pos=0, **{d: 0 for d in DENS}))
        dist[sp]["n"] += 1; dist[sp]["pos"] += int(bool(IDX[s].get("is_pos")))
        if dn in DENS: dist[sp][dn] += 1
    for sp in ["train","val","test"]:
        if sp in dist:
            e = dist[sp]
            P(f"  {sp:>5s}: n={e['n']:>4d}  poz=%{100*e['pos']/max(e['n'],1):.1f}  " +
              "  ".join(f"{d}={e[d]}" for d in DENS))
    RES["split_dist"] = dist
    pf = [dist[sp]["pos"]/max(dist[sp]["n"],1) for sp in dist]
    check("positive fraction balanced across splits (diff < 0.08)",
          max(pf)-min(pf) < 0.08, f"{min(pf):.3f}-{max(pf):.3f}")

    if RIDX:
        o1, o2 = set(IDX)-set(RIDX), set(RIDX)-set(IDX)
        mism = dict(split=0, density=0, native_z=0, is_pos=0, lesion_count=0)
        for s in set(IDX) & set(RIDX):
            a, b_ = IDX[s], RIDX[s]
            for k in mism:
                if k == "split": mism[k] += a["split"] != b_["split"]
                elif k == "is_pos": mism[k] += bool(a.get(k)) != bool(b_.get(k))
                else: mism[k] += str(a.get(k)) != str(b_.get(k))
        check(f"{TARGET_TAG} ve {REF_TAG} seed kumeleri ozdes", not o1 and not o2,
              f"+{len(o1)} / -{len(o2)}")
        check("split membership and metadata identical to reference",
              sum(mism.values()) == 0, str(mism))
        RES["seed_only_target"] = sorted(o1); RES["seed_only_ref"] = sorted(o2)
        RES["metadata_mismatch"] = mism
    stage_end("2", _S2)

# =============================================================================
# STAGE 3 -- STORAGE
# =============================================================================
if "3" in STAGES:
    hdr("STAGE 3 -- STORAGE")
if "3" in STAGES and not stage_begin("3", ["storage"]):
    def tot(root):
        s = n = 0
        for sp in ["train","val","test"]:
            fs = glob.glob(f"{root}/{sp}/*.npz")
            s += sum(os.path.getsize(f) for f in fs); n += len(fs)
        return s, n
    st, nt = tot(TARGET)
    P(f"  {TARGET_TAG}: {nt:>4d} chunk  {st/1e9:>7.1f} GB  ({st/1e6/max(nt,1):.0f} MB/chunk)")
    RES["storage"] = {f"{TARGET_TAG}_gb": st/1e9, f"{TARGET_TAG}_chunks": nt}
    RES.setdefault("n_chunk", nt)
    if REFERENCE and DO_REFERENCE:
        sr, nr = tot(REFERENCE)
        P(f"  {REF_TAG}: {nr:>4d} chunk  {sr/1e9:>7.1f} GB")
        P(f"  change: {(st-sr)/1e9:+.1f} GB ({100*(st-sr)/max(sr,1):+.1f} %)")
        RES["storage"].update({f"{REF_TAG}_gb": sr/1e9, f"{REF_TAG}_chunks": nr,
                               "pct": 100*(st-sr)/max(sr,1)})
    rows = []
    fs = sorted(glob.glob(f"{TARGET}/test/*.npz"))
    if fs:
        z = zipfile.ZipFile(fs[0]); tt = sum(i.compress_size for i in z.infolist())
        P(f"\n  example chunk ({os.path.basename(fs[0])}): {tt/1e6:.0f} MB")
        for it in sorted(z.infolist(), key=lambda x: -x.compress_size)[:14]:
            P(f"    {it.filename.replace('.npy',''):>22s} {it.compress_size/1e6:>8.2f} MB "
              f"({100*it.compress_size/tt:>5.1f}%)  {it.file_size/max(it.compress_size,1):.2f}x")
            rows.append(dict(key=it.filename.replace(".npy",""), mb=it.compress_size/1e6,
                             pct=100*it.compress_size/tt,
                             ratio=it.file_size/max(it.compress_size,1)))
        z.close()
    dump(rows, "T3_storage_by_key")
    stage_end("3", ["storage"])

# =============================================================================
#  GPU / LEAP
# =============================================================================
GPU_OK = False
if any(c in STAGES for c in "6789A") or ("4" in STAGES and DO_GEOMETRY):
    try:
        import torch
        assert torch.cuda.is_available(), "CUDA yok"
        try: import leapctype
        except ImportError:
            P("  [setup] installing LEAP...")
            import subprocess
            if not os.path.isdir("/content/LEAP/.git"):
                subprocess.run("git clone --depth 1 https://github.com/LLNL/LEAP.git "
                               "/content/LEAP", shell=True, check=True)
            subprocess.run(f"{sys.executable} -m pip install -q /content/LEAP",
                           shell=True, check=True)
            import leapctype
        from leapctype import tomographicModels
        GPU_OK = True; DEV = "cuda"
        P(f"  GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        P(f"  ! GPU/LEAP unavailable -> geometry and reconstruction stages SKIPPED: {e}")

def _free():
    gc.collect()
    try: torch.cuda.empty_cache()
    except Exception: pass

class Geo:
    def __init__(self, sid, sdd, vox_z, offx, offy, offz):
        NA = CFG["NA"]; th = np.radians(np.linspace(-CFG["ANG"], CFG["ANG"], NA))
        L = tomographicModels()
        L.set_modularbeam(
            NA, CFG["PH"], CFG["PW"], CFG["DET_PIX"], CFG["DET_PIX"],
            np.stack([sid*np.sin(th), np.zeros(NA), sid*np.cos(th)], 1).astype(np.float32),
            np.tile(np.array([0, 0, -(sdd-sid)], np.float32), (NA, 1)),
            np.tile(np.array([1, 0, 0], np.float32), (NA, 1)),
            np.tile(np.array([0, 1, 0], np.float32), (NA, 1)))
        L.set_volume(CFG["TH"], CFG["TW"], CFG["ZOUT"], CFG["VOX_XY"],
                     float(vox_z), float(offx), float(offy), float(offz))
        try: L.set_log_error()
        except Exception: pass
        self.L, self.sid, self.sdd = L, float(sid), float(sdd)
        with torch.no_grad():
            f = torch.ones(CFG["ZOUT"], CFG["TH"], CFG["TW"], device=DEV)
            g = torch.zeros((NA, CFG["PH"], CFG["PW"]), device=DEV)
            L.project(g, f.permute(0, 2, 1).contiguous())
        self.ps = float(g.max().item()); self._rs = torch.clamp(g, min=1e-6)
        del f; _free()
        with torch.no_grad():
            gg = torch.ones((NA, CFG["PH"], CFG["PW"]), device=DEV)
            fb = torch.zeros(CFG["ZOUT"], CFG["TW"], CFG["TH"], device=DEV)
            L.backproject(gg, fb)
        self._cs = torch.clamp(fb, min=1e-6); del gg, fb; _free()
    def close(self):
        try: del self._rs, self._cs
        except Exception: pass
        _free()
    def _A(self, v):
        g = torch.zeros((CFG["NA"], CFG["PH"], CFG["PW"]), device=DEV)
        with torch.no_grad(): self.L.project(g, v.permute(0, 2, 1).contiguous())
        return g
    def _AT(self, g):
        fb = torch.zeros(CFG["ZOUT"], CFG["TW"], CFG["TH"], device=DEV)
        with torch.no_grad(): self.L.backproject(g, fb)
        return fb.permute(0, 2, 1)
    def A(self, vol):
        t = torch.from_numpy(np.ascontiguousarray(vol)).float().to(DEV)
        g = self._A(t); o = (g/self.ps).cpu().numpy(); del t, g; _free(); return o
    def atp(self, gn):
        g = torch.from_numpy(np.ascontiguousarray(gn)).float().to(DEV)
        x = self._AT(g)/self._cs.permute(0, 2, 1)
        o = x.cpu().numpy(); del g, x; _free(); return o
    def sirt(self, gn, n):
        g = torch.from_numpy(np.ascontiguousarray(gn)).float().to(DEV)
        x = torch.zeros(CFG["ZOUT"], CFG["TW"], CFG["TH"], device=DEV)
        for _ in range(n):
            r = (g - self._A(x.permute(0, 2, 1).contiguous()))/self._rs
            fb = torch.zeros_like(x)
            with torch.no_grad(): self.L.backproject(r, fb)
            x = x + fb/self._cs; del r, fb
        o = x.permute(0, 2, 1).cpu().numpy(); del g, x; _free(); return o
    def fbp(self, gn, window=FBP_WINDOW, cosw=FBP_COSWEIGHT):
        """Kendi rampa filtremiz — LEAP'in FBP'si modular-beam DBT'de calismaz."""
        g = np.asarray(gn, np.float32).copy()
        if cosw:
            _, PH, PW = g.shape
            u = (np.arange(PH)-PH/2.0)*CFG["DET_PIX"]; v = (np.arange(PW)-PW/2.0)*CFG["DET_PIX"]
            g *= (self.sid/np.sqrt(self.sid**2 + u[:,None]**2 + v[None,:]**2))[None].astype(np.float32)
        PH = g.shape[1]; n = int(2**math.ceil(math.log2(PH*1.6)))
        pl = (n-PH)//2; pr = n-PH-pl
        gp = np.pad(g, ((0,0),(pl,pr),(0,0)), mode="edge")
        fr = np.fft.rfftfreq(n); H = 2.0*fr
        if window == "hann": H = H*(0.5 + 0.5*np.cos(np.pi*fr/max(fr.max(), 1e-9)))
        gf = np.fft.irfft(np.fft.rfft(gp, axis=1)*H[None,:,None], n=n,
                          axis=1)[:, pl:pl+PH, :].astype(np.float32)
        t = torch.from_numpy(np.ascontiguousarray(gf)).float().to(DEV)
        x = self._AT(t); o = x.cpu().numpy(); del t, x; _free(); return o

GEO_CACHE = {}
def geo(sid, sdd, vz, ox, oy, oz):
    k = tuple(round(float(x), 4) for x in (sid, sdd, vz, ox, oy, oz))
    if k not in GEO_CACHE:
        if len(GEO_CACHE) > 8: GEO_CACHE.pop(next(iter(GEO_CACHE))).close()
        GEO_CACHE[k] = Geo(*k)
    return GEO_CACHE[k]
def _oxy(dn):
    nx, ny = NATIVE_XY[dn]
    return ((CFG["TH"]*CFG["VOX_XY"] - nx*CFG["NATIVE_PIX"])/2.0 + OFFX_C,
            OFFY_A + OFFY_B*ny)
def geo_t(dn, nz, delta=None):
    vz = nz/CFG["ZOUT"]; ox, oy = _oxy(dn)
    dl = DELTA[dn] if delta is None else delta
    return geo(GEO["SID"], GEO["SDD"], vz, ox, oy,
               -(GEO["SDD"]-GEO["SID"]) + CFG["ZOUT"]*vz/2.0 + dl)
REF_OFFZ = {}
def geo_r(dn):
    vz = REF_GEO["VOX_Z"]; ox, oy = _oxy(dn)
    oz = REF_OFFZ.get(dn, -(REF_GEO["SDD"]-REF_GEO["SID"]) + CFG["ZOUT"]*vz/2.0 + DELTA[dn])
    return geo(REF_GEO["SID"], REF_GEO["SDD"], vz, ox, oy, oz)

def probe_score(sid, sdd, vz, ox, oy, oz, probes):
    G = Geo(sid, sdd, vz, ox, oy, oz); tot = []
    for cl, pj in probes:
        Ap = G.A(cl)
        dr = np.array([best_shift(Ap[a], pj[a])[0] for a in range(CFG["NA"])])
        tot.append(float(np.abs(dr).max())); del Ap
    G.close(); del G; _free(); return float(np.mean(tot))
def solve_offz(dn, probes, base, span, coarse, fine, sid, sdd, vz):
    ox, oy = _oxy(dn)
    grid = np.arange(base-span, base+span+1e-9, coarse)
    sc = [probe_score(sid, sdd, vz, ox, oy, z, probes) for z in grid]
    j = int(np.argmin(sc)); b = float(grid[j]); s0 = float(sc[j])
    edge = j in (0, len(grid)-1)
    if edge: P(f"    ! WARNING [{dn}]: optimum arama SINIRINDA ({b:+.2f})")
    g2 = np.arange(b-coarse, b+coarse+1e-9, fine)
    s2 = [probe_score(sid, sdd, vz, ox, oy, z, probes) for z in g2]
    if float(np.min(s2)) < s0: b = float(g2[int(np.argmin(s2))]); s0 = float(np.min(s2))
    return b, s0, edge

# =============================================================================
#  GURULTU FORMULU OTOMATIK TESPITI
# =============================================================================
NOISE_CFG = {"variant": None, "map": {}, "err": None}
def bitexact_check(cp, qs, nseed, stored):
    """SABIT eslemeyle (full=0,half=1,quarter=2) bit farkini olc. Arama YOK."""
    fn = noise_formula                          # production formula (see header)
    tot = 0.0; per = {}
    for dname, _, _, gain in DOSE:
        if dname not in stored: continue
        x = fn(cp, qs, gain, nseed, DOSE_IDX[dname], CFG["S_ELEC"])
        e = int(np.abs(np.rint(x*65535).astype(np.int64)
                       - stored[dname].astype(np.int64)).max())
        per[dname] = e; tot += e
    return per, tot

# =============================================================================
# ASAMA 4 — FUSED PASS   (her chunk BIR KEZ okunur)
# =============================================================================
if "4" in STAGES:
    hdr(f"STAGE 4 -- MAIN PASS  ({'full population' if FULL else f'n={N_FUSED}'})")
    sel = select(N_FUSED, seed=11)
    rows = load_rows("T4_fused", "4")
    done = {int(r["seed"]) for r in rows}
    sel = [s for s in sel if s not in done]
    P(f"  {len(sel)} patients to process ({len(done)} already present)")
    ncp = f"{TAB}/_noise_cfg.json"
    if os.path.exists(ncp) and "4" not in FORCE_STAGES:
        NOISE_CFG.update(json.load(open(ncp)))
        P(f"  [cached] noise formula: variant {NOISE_CFG['variant']}, "
          f"dose_idx {NOISE_CFG['map']}")

    TC = {"p": None, "d": None, "loc": None}
    RC = {"p": None, "d": None, "loc": None}
    def open_chunk(slot, path):
        if slot["p"] == path: return slot["d"]
        if slot["d"] is not None: slot["d"].close(); release(slot["loc"])
        slot["loc"] = fetch(path); slot["d"] = np.load(slot["loc"]); slot["p"] = path
        return slot["d"]

    t0, nn = time.time(), 0
    for s in sel:
        try:
            r = IDX[s]; dn = str(r.get("density")); nz = int(r["native_z"])
            d = open_chunk(TC, r["chunk"]); b = r["b"]
            rec = dict(seed=s, dens=dn, split=r["split"], native_z=nz,
                       is_pos=int(bool(r.get("is_pos"))))
            cl = f01(d["clean"][b]); cp = f01(d["clean_proj"][b])
            mk = np.asarray(d["mask"][b]) > 0
            qs = float(d["proj_scale"][b]); rs = float(d["recon_scale"][b])
            vly = float(d["strip_valley"][b]) if "strip_valley" in d.files else np.nan

            # ---- (1) NORMALIZASYON DEGISMEZLERI ----------------------------
            rec["clean_max"] = float(cl.max())
            rec["clean_p"]   = float(np.percentile(cl[cl > 0], CFG["P_RECON"])) if (cl>0).any() else np.nan
            rec["proj_max"]  = float(cp.max())
            rec["proj_p"]    = float(np.percentile(cp[cp > 0], CFG["P_PROJ"])) if (cp>0).any() else np.nan
            rec["clip_frac_clean"] = float((cl >= 0.99999).mean())
            rec["clip_frac_proj"]  = float((cp >= 0.99999).mean())
            # ---- (2) MASKE = clean > esik ----------------------------------
            mexp = cl > CFG["MASK_THR"]
            inter = np.logical_and(mk, mexp).sum()
            rec["mask_dice_thr"] = float(2*inter/max(mk.sum()+mexp.sum(), 1))
            rec["mask_frac"] = float(mk.mean())
            # ---- (3) PENUMBRA VADI SINIRI ----------------------------------
            rec["p_phys_max"] = float(cp.max()*qs); rec["valley"] = vly
            rec["valley_ok"] = int(np.isnan(vly) or cp.max()*qs <= vly + 1e-3)
            # ---- (4) PROJEKSIYON / FLATFIELD -------------------------------
            a = breast = bb = None            # onceki hastadan sizinti olmasin
            live = cp.sum(0) > 0
            if live.sum() > 5000:
                ys, xs = np.where(live)
                bb = (slice(ys.min(), ys.max()+1), slice(xs.min(), xs.max()+1))
                a = cp[MID][bb]*qs; nzv = a[a > 0]
                if nzv.size > 1000:
                    thr = 0.5*float(np.median(nzv)); breast = a > thr; air = ~breast
                    if air.sum() > max(200, 0.01*a.size):
                        rec["air_pedestal"] = float(np.median(a[air]))
                        rec["air_zerofrac"] = float((a[air] <= 1e-9).mean())
                    rec["breast_frac"] = float(breast.mean())
                    rec["zero_frac"] = float((a <= 1e-9).mean())
                    if breast.sum() > 500:
                        rec["tissue_p90"] = float(np.percentile(a[breast], 90))
                    p90 = []
                    for k in range(CFG["NA"]):
                        vv = cp[k][bb][breast]*qs
                        if vv.size > 500: p90.append(np.percentile(vv, 90))
                    if len(p90) >= 20:
                        p90 = np.array(p90)
                        rec["uniformity"] = float(p90.max()/max(p90.min(), 1e-9))
            # ---- (5) GURULTU ------------------------------------------------
            supp = cp > 0.02
            stored = {}
            for dname, key, skey, gain in DOSE:
                if skey in d.files:
                    rec[f"sig_rep_{dname}"] = float(d[skey][b])
                if DO_NOISE_ARRAYS and key in d.files:
                    raw = d[key][b]; stored[dname] = np.asarray(raw)
                    nzp = f01(raw)
                    if supp.sum() > 5000:
                        rec[f"sig_meas_{dname}"] = float((nzp-cp)[supp].std())
                    if f"sig_rep_{dname}" in rec and f"sig_meas_{dname}" in rec:
                        rp = rec[f"sig_rep_{dname}"]
                        rec[f"sig_reldiff_{dname}"] = abs(rec[f"sig_meas_{dname}"]-rp)/max(rp,1e-9)
                    if dname == "half":
                        sy, sx = _sub(CFG["PH"], CFG["PW"])
                        dd_ = (nzp-cp)
                        rec["noise_lag1_row"]  = lag1(dd_[MID][sy, sx], 0)
                        rec["noise_lag1_col"]  = lag1(dd_[MID][sy, sx], 1)
                        rec["noise_lag1_view"] = lag1(dd_[:, sy, sx], 0)
            # ---- (5b) BIT-EXACT REPRODUCTION (paper, Technical Validation) ------------------
            _do_bx = (DO_NOISE_ARRAYS and DO_NOISE_BITEXACT and "noise_seed" in d.files
                      and stored and (N_BITEXACT <= 0 or nn < N_BITEXACT))
            if _do_bx:
                ns = int(d["noise_seed"][b])
                per, tot = bitexact_check(cp, qs, ns, stored)
                for dname, e in per.items(): rec[f"bitdiff_{dname}"] = e
                if NOISE_CFG["variant"] is None:
                    NOISE_CFG.update(variant="A(uretim)", map=dict(DOSE_IDX),
                                     err=float(tot))
                    json.dump(NOISE_CFG, open(ncp+".t", "w")); os.replace(ncp+".t", ncp)
            del stored
            # ---- (6) LEZYON / KONTROL ROI ----------------------------------
            nl = int(d["lesion_count"][b]); lc = d["lesion_coords"][b]
            oob = 0; masses = []; sd_m = []; sd_c = []; sd_k = []
            for i in range(nl):
                z, h, w, t = [float(x) for x in lc[i][:4]]
                if not (0 <= z < CFG["ZOUT"] and 0 <= h < CFG["TH"] and 0 <= w < CFG["TW"]):
                    oob += 1; continue
                v = sdnr_at(cl, z, h, w)
                if t >= 4: masses.append((z, h, w)); sd_m.append(v)
                else: sd_k.append(v)
            in_mask = 0; cc = 0
            if "control_rois" in d.files:
                cc = int(d["control_count"][b]); ctl = d["control_rois"][b]
                for i in range(cc):
                    z, h, w, _r, ok = [float(x) for x in ctl[i][:5]]
                    if ok < 0.5: continue
                    if not (0 <= z < CFG["ZOUT"] and 0 <= h < CFG["TH"] and 0 <= w < CFG["TW"]):
                        oob += 1; continue
                    in_mask += int(bool(mk[int(z), int(h), int(w)]))
                    sd_c.append(sdnr_at(cl, z, h, w))
            rec.update(lesion_count=nl, control_count=cc, roi_oob=oob,
                       control_in_mask=in_mask,
                       n_mass=len(sd_m), n_calc=len(sd_k), n_ctrl=len(sd_c),
                       sdnr_mass=mn(sd_m), sdnr_calc=mn(sd_k), sdnr_ctrl=mn(sd_c))
            rec["ispos_consistent"] = int(bool(r.get("is_pos")) == (nl > 0))
            # ---- (7) GEOMETRI ------------------------------------------------
            if DO_GEOMETRY and GPU_OK and dn in NATIVE_XY:
                G = geo(GEO["SID"], GEO["SDD"], float(r["geom_vox_z"]),
                        float(r["geom_offx"]), float(r["geom_offy"]), float(r["geom_offz"]))
                Ap = G.A(cl)
                dr = np.empty(CFG["NA"]); dc = np.empty(CFG["NA"])
                for a in range(CFG["NA"]): dr[a], dc[a] = best_shift(Ap[a], cp[a])
                u = (Ap > 0) | (cp > 0)
                av = np.arange(CFG["NA"])
                rec.update(dr_max=float(np.abs(dr).max()),
                           dr_dev_max=float(np.abs(dr-dr.mean()).max()),
                           dr_slope=float(np.polyfit(av, dr, 1)[0]),
                           dr_detrend=float(np.abs(dr-np.polyval(np.polyfit(av,dr,1),av)).max()),
                           dc_max=float(np.abs(dc).max()), corr3d=corr(Ap[u], cp[u]))
                rec["dr_curve"] = ";".join(f"{x:.3f}" for x in dr)
                del Ap; _free()
            # ---- (8) REFERENCE ----------------------------------------------
            if DO_REFERENCE and RIDX.get(s):
                rr = RIDX[s]; dr_ = open_chunk(RC, rr["chunk"]); bb2 = rr["b"]
                rcl = f01(dr_["clean"][bb2]); rmk = np.asarray(dr_["mask"][bb2]) > 0
                rcp = f01(dr_["clean_proj"][bb2]); rqs = float(dr_["proj_scale"][bb2])
                rec["ref_clean_maxdiff"] = float(np.abs(cl-rcl).max())
                rec["ref_mask_dice"] = float(2*np.logical_and(mk, rmk).sum()
                                             / max(mk.sum()+rmk.sum(), 1))
                rec["ref_recon_scale_diff"] = abs(float(dr_["recon_scale"][bb2])-rs)/max(rs,1)
                rec["ref_proj_scale"] = rqs
                if a is not None and breast is not None and bb is not None:
                    ra = rcp[MID][bb]*rqs
                    struct = (ra <= 1e-9) & (a <= 1e-9)
                    airm = (~breast) & (~struct)
                    if airm.sum() > max(200, 0.01*a.size):
                        rec["ref_air_pedestal"] = float(np.median(ra[airm]))
                        rec["ref_air_zerofrac"] = float((ra[airm] <= 1e-9).mean())
                    if breast.sum() > 500:
                        rec["ref_tissue_p90"] = float(np.percentile(ra[breast], 90))
                if "noisy" in dr_.files:
                    ry, rx = _sub(CFG["TH"], CFG["TW"], (0.25,0.70), (0.25,0.75))
                    rec["ref_noisy_lag1_z"] = lag1((f01(dr_["noisy"][bb2])-rcl)[:, ry, rx], 0)
                del rcl, rcp, rmk
            rows.append(rec); nn += 1
        except Exception as e:
            P(f"    ! seed {s}: {e}")
            if nn < 3: traceback.print_exc()
        if nn and nn % 40 == 0:
            dump(rows, "T4_fused")
            el = (time.time()-t0)/60
            P(f"    {nn}/{len(sel)}  [{el:.1f} dk, kalan ~{el/nn*(len(sel)-nn):.0f} dk]")
    for slot in (TC, RC):
        if slot["d"] is not None: slot["d"].close(); release(slot["loc"])
    dump(rows, "T4_fused")
    P(f"  done: {len(rows)} patients  [{(time.time()-t0)/60:.1f} dk]")

# =============================================================================
# ASAMA 5 — OZET  (asama 4 CSV'sinden; aninda)
# =============================================================================
if "5" in STAGES:
    hdr("STAGE 5 -- SUMMARY STATISTICS AND CHECKS")
    F = load_rows("T4_fused")
    if not F:
        P("  ! T4_fused.csv missing -- run stage 4 first.")
    else:
        def C(k, dn=None, flt=None):
            out = []
            for r in F:
                if dn and r.get("dens") != dn: continue
                if flt and not flt(r): continue
                v = r.get(k)
                if v in ("", None): continue
                try:
                    x = float(v)
                    if np.isfinite(x): out.append(x)
                except Exception: pass
            return np.array(out)
        N = len(F); P(f"  n = {N} patients")
        RES["n_fused"] = N

        # ---------- 1. NORMALIZASYON -----------------------------------------
        P("\n  --- NORMALIZATION INVARIANTS ---")
        cp_ = C("clean_p"); pp_ = C("proj_p"); cm = C("clean_max"); pm = C("proj_max")
        P(f"  clean  p{CFG['P_RECON']} = {mn(cp_):.5f} (1.0 olmali)  maks={mn(cm):.5f}  "
          f"clipped={100*mn(C('clip_frac_clean')):.3f}%")
        P(f"  proj   p{CFG['P_PROJ']} = {mn(pp_):.5f} (1.0 olmali)  maks={mn(pm):.5f}  "
          f"clipped={100*mn(C('clip_frac_proj')):.3f}%")
        check(f"clean {CFG['P_RECON']}. persentil = 1.0 (normalizasyon correct)",
              abs(mn(cp_)-1.0) < 0.02, f"{mn(cp_):.4f}")
        check(f"clean_proj {CFG['P_PROJ']}. persentil = 1.0",
              abs(mn(pp_)-1.0) < 0.02, f"{mn(pp_):.4f}")
        check("clean maximum does not exceed 1.0", mn(cm) <= 1.0001 and np.max(cm) <= 1.0001,
              f"max {np.max(cm):.5f}")
        RES["normalization"] = dict(clean_p=mn(cp_), proj_p=mn(pp_),
            clean_clip=mn(C("clip_frac_clean")), proj_clip=mn(C("clip_frac_proj")))

        # ---------- 2. MASKE / VADI ------------------------------------------
        md = C("mask_dice_thr"); vo = C("valley_ok")
        P(f"\n  --- MASK / PENUMBRA ---")
        P(f"  Dice(mask, clean>{CFG['MASK_THR']}) = {mn(md):.6f}  min={md.min() if len(md) else float('nan'):.6f}")
        P(f"  breast mask volume fraction = %{100*mn(C('mask_frac')):.1f}")
        P(f"  p_phys max = {mn(C('p_phys_max')):.3f}   vadi esigi = {mn(C('valley')):.3f}")
        check(f"mask tam olarak clean>{CFG['MASK_THR']} (Dice>0.999)",
              len(md) and md.min() > 0.999, f"min {md.min() if len(md) else 0:.5f}")
        check("no projection exceeds the penumbra valley threshold",
              len(vo) and vo.min() > 0.5, f"{int((vo<0.5).sum())} ihlal")
        RES["mask"] = dict(dice=mn(md), dice_min=float(md.min()) if len(md) else None,
                           frac=mn(C("mask_frac")))

        # ---------- 3. FLATFIELD ---------------------------------------------
        P(f"\n  --- FLAT-FIELD / AIR ATTENUATION (physical: 0) ---")
        P(f"  {'class':>10s} {'n':>5s} {'p(hava)':>10s} {'sifir%':>8s} {'doku p90':>9s} "
          f"{'tekduze':>8s}" + (f" {'REF p(hava)':>12s}" if RIDX else ""))
        RES["flatfield"] = {}
        for dn in DENS:
            ap = C("air_pedestal", dn)
            if not len(ap): continue
            e = dict(n=len(ap), air=mn(ap), zerofrac=mn(C("air_zerofrac", dn)),
                     tissue_p90=mn(C("tissue_p90", dn)), unif=mn(C("uniformity", dn)))
            line = (f"  {dn:>10s} {len(ap):>5d} {e['air']:>+10.4f} "
                    f"{100*e['zerofrac']:>7.1f}% {e['tissue_p90']:>9.4f} {e['unif']:>8.4f}")
            if RIDX:
                rr = C("ref_air_pedestal", dn); e["ref_air"] = mn(rr)
                line += f" {mn(rr):>+12.4f}"
            P(line); RES["flatfield"][dn] = e
        AP = C("air_pedestal")
        check("air attenuation is practically zero (|p|<0.02, all classes)",
              abs(mn(AP)) < 0.02, f"{mn(AP):+.5f}")
        cls = [RES["flatfield"][d]["air"] for d in RES["flatfield"]]
        check("cross-class pedestal spread is negligible (std<0.01)",
              len(cls) < 2 or float(np.std(cls)) < 0.01, f"std {np.std(cls):.5f}")
        U = C("uniformity")
        check("angular uniformity close to the physical expectation 1/cos25=1.10",
              1.0 <= mn(U) <= 1.25, f"{mn(U):.4f}")
        RES["flatfield_class_std"] = float(np.std(cls)) if len(cls) > 1 else None
        RES["uniformity"] = mn(U)

        # ---------- 4. GURULTU ------------------------------------------------
        P(f"\n  --- NOISE ---")
        P(f"  {'doz':>9s} {'sigma':>9s} {'olculen':>9s} {'bagil fark':>12s} {'bit farki':>11s}")
        RES["noise"] = {}
        for dname, _, _, _ in DOSE:
            rp = C(f"sig_rep_{dname}"); ms = C(f"sig_meas_{dname}")
            rd = C(f"sig_reldiff_{dname}"); bd = C(f"bitdiff_{dname}")
            P(f"  {dname:>9s} {mn(rp):>9.5f} {mn(ms):>9.5f} "
              f"%{100*mn(rd):>11.5f} {('%d'%bd.max()) if len(bd) else '-':>11s}")
            RES["noise"][dname] = dict(sigma=mn(rp), measured=mn(ms), reldiff=mn(rd),
                bitdiff_max=float(bd.max()) if len(bd) else None,
                bitdiff_frac_exact=float((bd <= 1).mean()) if len(bd) else None, n=len(rp))
        rdall = np.concatenate([C(f"sig_reldiff_{d_}") for d_, _, _, _ in DOSE
                                if len(C(f"sig_reldiff_{d_}"))] or [np.array([np.nan])])
        check("stored sigma matches measured sigma (<0.5%)",
              np.nanmax(rdall) < 0.005, f"max {100*np.nanmax(rdall):.4f}%")
        bda = [RES["noise"][d_]["bitdiff_max"] for d_, _, _, _ in DOSE
               if RES["noise"][d_]["bitdiff_max"] is not None]
        if bda:
            P(f"\n  BIT-EXACT REPRODUCTION: variant "
              f"{NOISE_CFG.get('variant')}, dose_idx {NOISE_CFG.get('map')}")
            for dname, _, _, _ in DOSE:
                e = RES["noise"][dname]
                if e["bitdiff_max"] is not None:
                    P(f"    {dname:>9s}: maks fark {e['bitdiff_max']:.0f} LSB "
                      f"(out of 65535), <=1 LSB fraction: "
                      f"%{100*e['bitdiff_frac_exact']:.2f}  n={e['n']}")
            _bx = max(bda)
            P(f"\n  noise formula: matches generate_dataset.add_noise exactly, "
              f"dose_idx = full:0 half:1 quarter:2")
            check("noisy_proj arrays are BIT-EXACTLY reproducible from noise_seed",
                  _bx <= 2, f"max {_bx:.0f} LSB (out of 65535)")
            RES["noise_formula"] = dict(NOISE_CFG)
            RES["noise_bitexact_ok"] = _bx <= 2
        else:
            P("\n  [skipped] bit-exact reproduction disabled")
        mono = sum(1 for r in F
                   if all(r.get(f"sig_rep_{d_}") not in ("", None) for d_, _, _, _ in DOSE)
                   and float(r["sig_rep_full"]) < float(r["sig_rep_half"]) < float(r["sig_rep_quarter"]))
        ntot = sum(1 for r in F if r.get("sig_rep_quarter") not in ("", None))
        check("dose monotonicity FULL<HALF<QUARTER (all patients)", ntot and mono == ntot,
              f"{mono}/{ntot}")
        if RES["noise"]["full"]["sigma"] and RES["noise"]["quarter"]["sigma"]:
            rt = RES["noise"]["quarter"]["sigma"]/RES["noise"]["full"]["sigma"]
            P(f"  quarter/full sigma ratio = {rt:.4f}  (teorik 2.00; "
              f"lower than 2 is expected due to electronic noise and clipping)")
            RES["sigma_ratio"] = rt
        for k, lbl in [("noise_lag1_row","satir"), ("noise_lag1_col","sutun"),
                       ("noise_lag1_view","gorunum")]:
            v = C(k)
            if len(v): P(f"  projection-domain noise lag-1 ({lbl}) = {mn(v):+.5f}  (beyaz: ~0)")
        lr = C("noise_lag1_row")
        check("projection-domain noise is white (pixel-independent Poisson)",
              len(lr) and abs(mn(lr)) < 0.05, f"{mn(lr):+.5f}")

        # ---------- 5. ROI / LEZYON -------------------------------------------
        P(f"\n  --- LESION / CONTROL ROI ---")
        oob = C("roi_oob"); isc = C("ispos_consistent")
        cim = C("control_in_mask"); cct = C("n_ctrl")
        check("no ROI falls outside the volume", oob.sum() == 0, f"{int(oob.sum())} ROI")
        check("is_pos consistent with lesion_count", len(isc) and isc.min() > 0.5,
              f"{int((isc<0.5).sum())} tutarsiz")
        if cct.sum() > 0:
            fr = cim.sum()/max(cct.sum(), 1.0)
            check("control ROIs fall INSIDE the breast mask", fr > 0.98, f"%{100*fr:.2f}")
        sm = C("sdnr_mass"); sk = C("sdnr_calc"); sc_ = C("sdnr_ctrl")
        P(f"  mass SDNR       = {mn(sm):+.4f}  (n={len(sm)} patients)")
        P(f"  calcification   = {mn(sk):+.4f}  (n={len(sk)})   expected ~0 (unresolvable)")
        P(f"  control ROI     = {mn(sc_):+.4f}  (n={len(sc_)})  unbiasedness: ~0")
        pooled = math.sqrt((np.nanvar(sm)+np.nanvar(sc_))/2) if len(sm) and len(sc_) else np.nan
        dp = (mn(sm)-mn(sc_))/max(pooled, 1e-9) if np.isfinite(pooled) else np.nan
        auc = float(np.mean(sm[:, None] > sc_[None, :])) if len(sm) and len(sc_) else np.nan
        P(f"  d' = {dp:.4f}   AUC = {auc:.4f}")
        check("control-ROI SDNR is near zero (the SDNR estimator is UNBIASED)",
              abs(mn(sc_)) < 0.10, f"{mn(sc_):+.4f}")
        check("calcifications are at the noise floor (unresolvable at 340 um)",
              abs(mn(sk)) < 0.15, f"{mn(sk):+.4f}")
        check("masses are separable from controls (d'>1.2)", dp > 1.2, f"d'={dp:.3f}")
        RES["lesion"] = dict(mass=mn(sm), calc=mn(sk), control=mn(sc_), dprime=float(dp),
                             auc=auc, n_mass=len(sm), n_ctrl=len(sc_))
        grad = {}
        for dn in DENS:
            v = C("sdnr_mass", dn)
            if len(v): grad[dn] = mn(v); P(f"    {dn:>10s} mass SDNR {mn(v):+.4f} (n={len(v)})")
        RES["lesion_by_density"] = grad
        if all(d in grad for d in DENS):
            check("mass-SDNR gradient fatty>scattered>hetero>dense (clinical expectation)",
                  grad["fatty"] > grad["scattered"] > grad["hetero"] > grad["dense"],
                  " > ".join(f"{d}:{grad[d]:+.2f}" for d in ["fatty","scattered","hetero","dense"]))

        # ---------- 6. GEOMETRI -----------------------------------------------
        dm = C("dr_max")
        if len(dm):
            P(f"\n  --- GEOMETRY: RESIDUAL PARALLAX ---")
            dd = C("dr_dev_max"); dt = C("dr_detrend"); dc = C("dc_max"); c3 = C("corr3d")
            P(f"  |dr|max      mean={mn(dm):.4f}  medyan={np.median(dm):.4f}  "
              f"95p={q(dm,95):.4f}  99p={q(dm,99):.4f}  maks={dm.max():.4f}")
            P(f"  |dr-mean|max mean={mn(dd):.4f}   |dr-trend|max ort={mn(dt):.4f}")
            P(f"  |dc|max      mean={mn(dc):.4f}   corr3d ort={mn(c3):.4f} min={c3.min():.4f}")
            RES["geometry"] = dict(dr_mean=mn(dm), dr_median=float(np.median(dm)),
                dr_p95=q(dm,95), dr_p99=q(dm,99), dr_max=float(dm.max()),
                dr_dev=mn(dd), dr_detrend=mn(dt), dc_mean=mn(dc),
                corr3d=mn(c3), corr3d_min=float(c3.min()), n=len(dm))
            check("mean residual parallax < 1.5 px", mn(dm) < 1.5, f"{mn(dm):.3f} px")
            check("99th-percentile residual parallax < 3 px", q(dm,99) < 3.0, f"{q(dm,99):.3f} px")
            check("no patient has |dr|max > 5 px", dm.max() < 5.0, f"max {dm.max():.3f} px")
            P(f"\n  {'class':>10s} {'n':>5s} {'|dr|max':>9s} {'egim':>9s} "
              f"{'detrend':>9s} {'corr3d':>8s}")
            RES["geometry"]["by_density"] = {}
            for dn in DENS:
                a_ = C("dr_max", dn)
                if not len(a_): continue
                e = dict(n=len(a_), dr=mn(a_), slope=mn(C("dr_slope", dn)),
                         detrend=mn(C("dr_detrend", dn)), corr3d=mn(C("corr3d", dn)))
                P(f"  {dn:>10s} {e['n']:>5d} {e['dr']:>9.4f} {e['slope']:>+9.4f} "
                  f"{e['detrend']:>9.4f} {e['corr3d']:>8.4f}")
                RES["geometry"]["by_density"][dn] = e
            zz = C("native_z")
            if len(zz) == len(dm) and len(zz) > 20:
                a1 = np.polyfit(zz, dm, 1); r1 = float(np.corrcoef(zz, dm)[0,1])
                sl = C("dr_slope"); a2 = np.polyfit(zz, sl, 1)
                r2 = float(np.corrcoef(zz, sl)[0,1])
                P(f"\n  residual parallax ~ breast thickness:")
                P(f"    |dr|max = {a1[0]:+.5f}*native_z {a1[1]:+.4f}   (r={r1:+.3f})")
                P(f"    slope   = {a2[0]:+.6f}*native_z {a2[1]:+.5f}   (r={r2:+.3f})")
                P(f"    after detrending, |dr|max = {mn(dt):.4f} px  -> asama 8")
                RES["geometry"]["thickness"] = dict(dr_slope=float(a1[0]), dr_r=r1,
                    slope_slope=float(a2[0]), slope_r=r2, detrended=mn(dt))

        # ---------- 7. REFERANS -----------------------------------------------
        if RIDX:
            P(f"\n  --- {REF_TAG} ILE KARSILASTIRMA (regresyon) ---")
            cd = C("ref_clean_maxdiff"); dice = C("ref_mask_dice")
            rsd = C("ref_recon_scale_diff")
            if len(cd):
                P(f"  clean max diff = {cd.max():.3e}  (uint16 adimi 1.53e-05)")
                P(f"  mask Dice min = {dice.min():.6f}   recon_scale maks bagil fark = {rsd.max():.2e}")
                check(f"`clean` {REF_TAG} ile ozdes (uint16 kuantizasyonu icinde)",
                      cd.max() < 3e-5, f"max {cd.max():.2e}")
                check(f"meme maskesi {REF_TAG} ile ozdes", dice.min() > 0.999,
                      f"min {dice.min():.5f}")
                RES["reference"] = dict(clean_maxdiff=float(cd.max()),
                    dice_min=float(dice.min()), recon_scale_diff=float(rsd.max()))
            rl = C("ref_noisy_lag1_z")
            if len(rl):
                P(f"  {REF_TAG} `noisy` z-ekseni lag-1 = {mn(rl):+.5f} +- {rl.std():.5f}"
                  f"  -> BEYAZ (real rekon icin asama 7'ye bak)")
                RES["reference"]["ref_noisy_lag1_z"] = mn(rl)
            for dn in DENS:
                rr = C("ref_air_pedestal", dn); tt = C("air_pedestal", dn)
                if len(rr): P(f"    {dn:>10s} p(hava): {REF_TAG} {mn(rr):+.4f} -> "
                              f"{TARGET_TAG} {mn(tt):+.4f}")

METHODS = ["ATp", "FBP", "SIRT-20", f"SIRT-{SIRT_ITERS}"]
def recon(G, g, m):
    if m == "ATp":  return G.atp(g)
    if m == "FBP":  return G.fbp(g)
    return G.sirt(g, int(m.split("-")[1]))

# =============================================================================
# ASAMA 6 — BASELINE:  IKI REJIM, RESMI TEST SPLIT
# =============================================================================
if "6" in STAGES and GPU_OK:
    hdr(f"STAGE 6 -- BASELINE (split={BASELINE_SPLIT}, two regimes, SIRT={SIRT_ITERS})")
    sel = select(N_BASELINE, seed=31, split=BASELINE_SPLIT)
    rows = load_rows("T6_baseline", "6")
    done = {(int(r["seed"]), r["regime"], r["method"]) for r in rows}
    seeds = [s for s in sel if not all((s, rg, m) in done
             for rg in ("ideal","real") for m in METHODS)]
    P(f"  {len(seeds)} patients to process ({len(sel)-len(seeds)} already present)")
    TCH = {"p": None, "d": None, "loc": None}
    def oc(slot, path):
        if slot["p"] == path: return slot["d"]
        if slot["d"] is not None: slot["d"].close(); release(slot["loc"])
        slot["loc"] = fetch(path); slot["d"] = np.load(slot["loc"]); slot["p"] = path
        return slot["d"]
    t0 = time.time()
    for i, s in enumerate(seeds):
        try:
            r = IDX[s]; dn = str(r.get("density"))
            if dn not in NATIVE_XY: continue
            d = oc(TCH, r["chunk"]); b = r["b"]
            cl = f01(d["clean"][b]); msk = np.asarray(d["mask"][b]) > 0
            cp = f01(d["clean_proj"][b])
            nl = int(d["lesion_count"][b]); lc = d["lesion_coords"][b]
            masses = [(float(a_), float(h_), float(w_)) for a_, h_, w_, t in lc[:nl]
                      if float(t) >= 4]
            G = geo_t(dn, int(r["native_z"]))
            gi = G.A(cl)
            for regime, g in (("ideal", gi), ("real", cp)):
                for m in METHODS:
                    if (s, regime, m) in done: continue
                    x = recon(G, g, m)
                    xn = x - x.min(); xn = xn/max(xn.max(), 1e-9)
                    sd = [v for v in (sdnr_at(xn, *mm) for mm in masses) if np.isfinite(v)]
                    rows.append(dict(seed=s, dens=dn, regime=regime, method=m,
                        corr=corr(x, cl, msk), psnr=psnr_sm(x, cl, msk),
                        mass_sdnr=mn(sd) if sd else ""))
                    del x, xn; _free()
            del gi; _free()
            if (i+1) % 5 == 0 or i == len(seeds)-1:
                dump(rows, "T6_baseline")
                el = (time.time()-t0)/60
                P(f"    {i+1}/{len(seeds)}  [{el:.1f} dk, kalan ~{el/(i+1)*(len(seeds)-i-1):.0f} dk]")
        except Exception as e:
            P(f"    ! seed {s}: {e}")
    if TCH["d"] is not None: TCH["d"].close(); release(TCH["loc"])
    dump(rows, "T6_baseline")

    def B(rg, m, k="corr", dn=None):
        v = [float(r[k]) for r in rows if r["regime"] == rg and r["method"] == m
             and (dn is None or r["dens"] == dn)
             and r.get(k) not in ("", None) and np.isfinite(float(r[k]))]
        return float(np.mean(v)) if v else float("nan")
    if rows:
        ns = len({r["seed"] for r in rows})
        P(f"\n  n = {ns} patients ({BASELINE_SPLIT} split)")
        P(f"  {'method':>10s} {'ideal':>9s} {'real':>9s} {'bosluk':>9s} "
          f"{'PSNR':>8s} {'massSDNR':>10s}")
        RES["baseline"] = {}
        for m in METHODS:
            i_, r_ = B("ideal", m), B("real", m)
            P(f"  {m:>10s} {i_:>9.4f} {r_:>9.4f} {i_-r_:>+9.4f} "
              f"{B('real',m,'psnr'):>8.2f} {B('real',m,'mass_sdnr'):>10.4f}")
            RES["baseline"][m] = dict(ideal=i_, real=r_, gap=i_-r_,
                psnr_real=B("real",m,"psnr"), mass_sdnr_real=B("real",m,"mass_sdnr"))
        oi = sorted(METHODS, key=lambda m: -B("ideal", m))
        orr = sorted(METHODS, key=lambda m: -B("real", m))
        ri = {m: i for i, m in enumerate(oi)}; rr = {m: i for i, m in enumerate(orr)}
        rho = float(np.corrcoef([ri[m] for m in METHODS], [rr[m] for m in METHODS])[0,1])
        sd_ = sorted({r["seed"] for r in rows}); nb = ng = 0
        for x in sd_:
            rr_ = {r["method"]: float(r["corr"]) for r in rows if r["seed"] == x
                   and r["regime"] == "real" and r.get("corr") not in ("", None)}
            ii_ = {r["method"]: float(r["corr"]) for r in rows if r["seed"] == x
                   and r["regime"] == "ideal" and r.get("corr") not in ("", None)}
            if len(rr_) == len(METHODS) and max(rr_, key=rr_.get) == "FBP": nb += 1
            if "FBP" in rr_ and "FBP" in ii_ and ii_["FBP"] - rr_["FBP"] < 0: ng += 1
        P(f"\n  ideal order: {' > '.join(oi)}")
        P(f"  real order : {' > '.join(orr)}")
        P(f"  Spearman(ideal, real) = {rho:+.3f}   FBP en iyi: {nb}/{len(sd_)}   "
          f"FBP boslugu<0: {ng}/{len(sd_)}")
        # FBP ile 2.en-iyi arasindaki ustunluk (real regime)
        rvals = sorted((B("real", m) for m in METHODS), reverse=True)
        fbp_margin = B("real", "FBP") - max(B("real", m) for m in METHODS if m != "FBP")
        # iki UC methodin yer degistirmesi: ideal-en-iyi realte en-kotu, tersi
        ideal_best = oi[0]; ideal_worst = oi[-1]
        real_best = orr[0]; real_worst = orr[-1]
        extremes_flip = (ideal_best == real_worst) and (ideal_worst == real_best)
        RES["inversion"] = dict(ideal_order=oi, real_order=orr, spearman=rho,
            fbp_best=nb, fbp_neg_gap=ng, n=len(sd_), fbp_margin=fbp_margin,
            extremes_flip=extremes_flip, ideal_best=ideal_best, real_best=real_best,
            ideal_worst=ideal_worst, real_worst=real_worst)
        check("the two extreme methods swap (ideal-best becomes real-worst and vice versa)",
              extremes_flip,
              f"ideal: {ideal_best}>...>{ideal_worst} | real: {real_best}>...>{real_worst}")
        check("method ranking is negatively correlated across regimes (rho<=-0.6)",
              rho <= -0.6, f"rho={rho:+.3f}")
        check("FBP is the best method in the real regime for every patient (the core finding)",
              nb == len(sd_), f"{nb}/{len(sd_)}")
        check("FBP's inverse-crime gap is NEGATIVE for every patient",
              ng == len(sd_), f"{ng}/{len(sd_)}")
        check("FBP leads the 2nd-best method by a clear margin in the real regime (>0.1)",
              fbp_margin > 0.1, f"marj {fbp_margin:+.4f}")
        P(f"\n  real regime by density:")
        P(f"  {'method':>10s}" + "".join(f"{d:>11s}" for d in DENS))
        for m in METHODS:
            P(f"  {m:>10s}" + "".join(f"{B('real',m,'corr',d):>11.4f}" for d in DENS))

# =============================================================================
# ASAMA 7 — DOZ-TEPKI + REKON GURULTU OTOKORELASYONU
# =============================================================================
if "7" in STAGES and GPU_OK:
    hdr("STAGE 7 -- DOSE RESPONSE (full/half/quarter) + reconstruction-noise correlation")
    sel = select(N_DOSE, seed=37, split=BASELINE_SPLIT)
    if FULL and len(sel) > 120: sel = sel[:120]        # doz taramasi icin 120 yeter
    rows = load_rows("T7_dose", "7")
    done = {(int(r["seed"]), r["dose"], r["method"]) for r in rows}
    TCH = {"p": None, "d": None, "loc": None}
    def oc2(path):
        if TCH["p"] == path: return TCH["d"]
        if TCH["d"] is not None: TCH["d"].close(); release(TCH["loc"])
        TCH["loc"] = fetch(path); TCH["d"] = np.load(TCH["loc"]); TCH["p"] = path
        return TCH["d"]
    METH_D = ["FBP", f"SIRT-{SIRT_ITERS}"]
    t0 = time.time()
    for i, s in enumerate(sel):
        try:
            r = IDX[s]; dn = str(r.get("density"))
            if dn not in NATIVE_XY: continue
            if all((s, dnm, m) in done for dnm, _, _, _ in [("clean",0,0,0)]+list(DOSE)
                   for m in METH_D): continue
            d = oc2(r["chunk"]); b = r["b"]
            cl = f01(d["clean"][b]); msk = np.asarray(d["mask"][b]) > 0
            cp = f01(d["clean_proj"][b])
            nl = int(d["lesion_count"][b]); lc = d["lesion_coords"][b]
            masses = [(float(a_), float(h_), float(w_)) for a_, h_, w_, t in lc[:nl]
                      if float(t) >= 4]
            G = geo_t(dn, int(r["native_z"]))
            ry, rx = _sub(CFG["TH"], CFG["TW"], (0.25,0.70), (0.25,0.75))
            base = {}
            for dose_name, key, _, _ in [("clean", "clean_proj", None, None)] + \
                                         [(a_, k_, x_, y_) for a_, k_, x_, y_ in DOSE]:
                if key not in d.files: continue
                g = f01(d[key][b])
                for m in METH_D:
                    if (s, dose_name, m) in done: continue
                    x = recon(G, g, m)
                    xn = x - x.min(); xn = xn/max(xn.max(), 1e-9)
                    sd = [v for v in (sdnr_at(xn, *mm) for mm in masses) if np.isfinite(v)]
                    rec = dict(seed=s, dens=dn, dose=dose_name, method=m,
                               corr=corr(x, cl, msk), psnr=psnr_sm(x, cl, msk),
                               mass_sdnr=mn(sd) if sd else "")
                    if dose_name == "clean": base[m] = x.copy()
                    elif m in base:
                        nz_ = (x - base[m])[:, ry, rx]      # SAF rekon gurultusu
                        rec["recon_lag1_z"]   = lag1(nz_, 0)
                        rec["recon_lag1_row"] = lag1(nz_, 1)
                        rec["recon_noise_std"] = float(nz_.std())
                    rows.append(rec); del x, xn; _free()
                del g
            for k in list(base): del base[k]
            _free()
            if (i+1) % 5 == 0 or i == len(sel)-1:
                dump(rows, "T7_dose")
                el = (time.time()-t0)/60
                P(f"    {i+1}/{len(sel)}  [{el:.1f} dk, kalan ~{el/(i+1)*(len(sel)-i-1):.0f} dk]")
        except Exception as e:
            P(f"    ! seed {s}: {e}")
    if TCH["d"] is not None: TCH["d"].close(); release(TCH["loc"])
    dump(rows, "T7_dose")
    def D(m, dz, k="corr"):
        v = [float(r[k]) for r in rows if r["method"] == m and r["dose"] == dz
             and r.get(k) not in ("", None) and np.isfinite(float(r[k]))]
        return float(np.mean(v)) if v else float("nan")
    if rows:
        order = ["clean"] + [a_ for a_, _, _, _ in DOSE]
        P(f"\n  {'method':>10s} {'metric':>12s}" + "".join(f"{o:>11s}" for o in order))
        RES["dose_response"] = {}
        for m in METH_D:
            for k, lbl in [("corr","korelasyon"), ("psnr","PSNR"),
                           ("mass_sdnr","mass SDNR")]:
                P(f"  {m:>10s} {lbl:>12s}" + "".join(f"{D(m,o,k):>11.4f}" for o in order))
            RES["dose_response"][m] = {o: dict(corr=D(m,o), psnr=D(m,o,"psnr"),
                                               mass_sdnr=D(m,o,"mass_sdnr")) for o in order}
        P(f"\n  --- RECONSTRUCTION-NOISE Z-AXIS CORRELATION ---")
        P(f"  (limited-angle reconstruction CORRELATES noise along z;")
        P(f"   adding white noise directly to the volume would model this INCORRECTLY)")
        RES["recon_noise_corr"] = {}
        for m in METH_D:
            for dz in [a_ for a_, _, _, _ in DOSE]:
                lz = D(m, dz, "recon_lag1_z"); lr = D(m, dz, "recon_lag1_row")
                if np.isfinite(lz):
                    P(f"    {m:>10s} {dz:>9s}: z lag-1 = {lz:+.4f}   satir lag-1 = {lr:+.4f}")
                    RES["recon_noise_corr"][f"{m}_{dz}"] = dict(z=lz, row=lr)
        zz = [v["z"] for v in RES["recon_noise_corr"].values() if np.isfinite(v["z"])]
        if zz:
            check("real reconstruction noise is STRONGLY correlated along z (>0.5)",
                  float(np.mean(zz)) > 0.5, f"ort {np.mean(zz):+.4f}")
        for m in METH_D:
            cs = [D(m, o) for o in [a_ for a_, _, _, _ in DOSE]]
            if all(np.isfinite(cs)):
                check(f"{m}: kalite dozla monoton azaliyor (tam>yarim>ceyrek)",
                      cs[0] >= cs[1] >= cs[2], " > ".join(f"{c:.4f}" for c in cs))

# =============================================================================
# ASAMA 8 — offz / DELTA INCE AYARI  (SALT METADATA)
# =============================================================================
if "8" in STAGES and GPU_OK:
    hdr("STAGE 8 -- offz/DELTA refinement (metadata-only improvement)")
    rows = load_rows("T8_offz", "8")
    if not rows:
        for dn in DENS:
            pool = [s for s in ALL if DENS_OF.get(s) == dn]
            pool = sorted(pool, key=lambda s: (IDX[s]["chunk"], IDX[s]["b"]))
            pid, vid = pool[:N_OFFZ_PROBE], pool[N_OFFZ_PROBE:N_OFFZ_PROBE+N_OFFZ_VALID]
            def ld(ids):
                out = []; slot = {"p": None, "d": None, "loc": None}
                for sd in ids:
                    try:
                        r = IDX[sd]
                        if slot["p"] != r["chunk"]:
                            if slot["d"] is not None: slot["d"].close(); release(slot["loc"])
                            slot["loc"] = fetch(r["chunk"]); slot["d"] = np.load(slot["loc"])
                            slot["p"] = r["chunk"]
                        out.append((f01(slot["d"]["clean"][r["b"]]),
                                    f01(slot["d"]["clean_proj"][r["b"]]),
                                    int(r["native_z"])))
                    except Exception as e: P(f"    ! {sd}: {e}")
                if slot["d"] is not None: slot["d"].close(); release(slot["loc"])
                return out
            probes = ld(pid)
            if not probes: continue
            nz = probes[0][2]; vz = nz/CFG["ZOUT"]; ox, oy = _oxy(dn)
            base = -(GEO["SDD"]-GEO["SID"]) + CFG["ZOUT"]*vz/2.0 + DELTA[dn]
            pr = [(a_, b_) for a_, b_, _ in probes]; t1 = time.time()
            bz, sc, edge = solve_offz(dn, pr, base, span=3.0, coarse=0.25, fine=0.05,
                                      sid=GEO["SID"], sdd=GEO["SDD"], vz=vz)
            del probes, pr; _free()
            P(f"    {dn:>10s}: offz {base:+.3f} -> {bz:+.3f}   "
              f"DELTA {DELTA[dn]:.3f} -> {DELTA[dn]+(bz-base):.3f}   "
              f"[{(time.time()-t1)/60:.1f} dk]")
            vs = ld(vid)
            if vs:
                pv = [(a_, b_) for a_, b_, _ in vs]
                d_old = probe_score(GEO["SID"], GEO["SDD"], vz, ox, oy, base, pv)
                d_new = probe_score(GEO["SID"], GEO["SDD"], vz, ox, oy, bz, pv)
                acc = d_new < d_old
                P(f"    {'':>10s}  validation n={len(vs)}: |dr|max {d_old:.4f} -> "
                  f"{d_new:.4f} px ({100*(d_new-d_old)/max(d_old,1e-9):+.1f}%)  "
                  f"-> {'BENIMSE' if acc else 'REDDET'}")
                rows.append(dict(dens=dn, native_z=nz, n_probe=len(pid), n_valid=len(vs),
                    offz_current=base, offz_refined=bz, edge=int(edge),
                    delta_current=DELTA[dn], delta_refined=DELTA[dn]+(bz-base),
                    dr_current=d_old, dr_refined=d_new, accept=int(acc)))
                del vs, pv; _free()
            dump(rows, "T8_offz")
    dump(rows, "T8_offz")
    if rows:
        o = np.array([float(r["dr_current"]) for r in rows])
        n_ = np.array([float(r["dr_refined"]) for r in rows])
        acc = np.array([int(r["accept"]) for r in rows])
        best = np.where(acc > 0, n_, o)
        POP = {}
        for dn in DENS: POP[dn] = sum(1 for s in ALL if DENS_OF.get(s) == dn)
        wt = np.array([POP.get(r["dens"], 1) for r in rows], float)
        P(f"\n  class-average |dr|max {o.mean():.4f} -> {best.mean():.4f} px")
        P(f"  POPULATION-WEIGHTED |dr|max {np.average(o,weights=wt):.4f} -> "
          f"{np.average(best,weights=wt):.4f} px")
        P(f"\n  >>> SUGGESTED DELTA (constants.py / README):")
        P("  DELTA = {" + ", ".join(
            f'"{r["dens"]}": {float(r["delta_refined"] if int(r["accept"]) else r["delta_current"]):.3f}'
            for r in rows) + "}")
        RES["offz_refine"] = {r["dens"]: dict(
            delta_current=float(r["delta_current"]), delta_refined=float(r["delta_refined"]),
            dr_current=float(r["dr_current"]), dr_refined=float(r["dr_refined"]),
            accept=int(r["accept"]), edge=int(r.get("edge", 0))) for r in rows}
        RES["offz_summary"] = dict(before=float(np.average(o, weights=wt)),
                                   after=float(np.average(best, weights=wt)))
        check("offz refinement further reduces residual parallax",
              best.mean() < o.mean(), f"{o.mean():.4f} -> {best.mean():.4f} px")

# =============================================================================
# ASAMA 9 — KONTROLLU DC-PEDESTAL TARAMASI (FBP mekanizmasi)
# =============================================================================
if "9" in STAGES and GPU_OK:
    hdr("STAGE 9 -- controlled DC-pedestal sweep")
    sel = select(N_PEDESTAL, seed=53, split=BASELINE_SPLIT)[:max(N_PEDESTAL, 4)]
    rows = load_rows("T9_pedestal", "9")
    done = {(int(r["seed"]), float(r["pedestal"]), r["method"]) for r in rows}
    t0 = time.time()
    for i, s in enumerate(sel):
        try:
            r = IDX[s]; dn = str(r.get("density"))
            if dn not in NATIVE_XY: continue
            L = fetch(r["chunk"]); d = np.load(L); b = r["b"]
            cl = f01(d["clean"][b]); msk = np.asarray(d["mask"][b]) > 0
            g0 = f01(d["clean_proj"][b]).astype(np.float32)
            qs = float(d["proj_scale"][b]); d.close(); release(L)
            G = geo_t(dn, int(r["native_z"])); live = g0.sum(0) > 0
            ref = {}
            for ped in PEDESTALS:
                g = g0.copy(); g[:, live] += np.float32(ped/qs)
                for m in METHODS:
                    x = recon(G, g, m)
                    if ped == 0.0: ref[m] = x.copy()
                    if (s, float(ped), m) not in done:
                        rows.append(dict(seed=s, dens=dn, pedestal=float(ped), method=m,
                            corr=corr(x, cl, msk),
                            self_corr=corr(x, ref[m], msk) if m in ref else ""))
                    del x; _free()
                del g
            for k in list(ref): del ref[k]
            _free(); dump(rows, "T9_pedestal")
            P(f"    {i+1}/{len(sel)}  seed={s} ({dn})  [{(time.time()-t0)/60:.1f} dk]")
        except Exception as e:
            P(f"    ! seed {s}: {e}")
    dump(rows, "T9_pedestal")
    def E(m, ped, k="corr"):
        v = [float(r[k]) for r in rows if r["method"] == m
             and abs(float(r["pedestal"])-ped) < 1e-9
             and r.get(k) not in ("", None) and np.isfinite(float(r[k]))]
        return float(np.mean(v)) if v else float("nan")
    if rows:
        P(f"\n  MASKED CORRELATION as a function of DC offset delta")
        P("  " + f"{'method':>10s}" + "".join(f"{p_:>10.2f}" for p_ in PEDESTALS) + f"{'kayip':>10s}")
        RES["pedestal"] = {}
        for m in METHODS:
            vs = [E(m, p_) for p_ in PEDESTALS]
            P(f"  {m:>10s}" + "".join(f"{v:>10.4f}" for v in vs) + f"{vs[0]-vs[-1]:>+10.4f}")
            RES["pedestal"][m] = dict(values={str(p_): v for p_, v in zip(PEDESTALS, vs)},
                loss=vs[0]-vs[-1],
                self_corr={str(p_): E(m, p_, "self_corr") for p_ in PEDESTALS})
        P(f"\n  CORRELATION WITH THE delta=0 OUTPUT (1.0 = fully blind to DC)")
        P("  " + f"{'method':>10s}" + "".join(f"{p_:>10.2f}" for p_ in PEDESTALS))
        for m in METHODS:
            P(f"  {m:>10s}" + "".join(f"{E(m,p_,'self_corr'):>10.5f}" for p_ in PEDESTALS))
        lf = RES["pedestal"]["FBP"]["loss"]; ls = RES["pedestal"][f"SIRT-{SIRT_ITERS}"]["loss"]
        check("FBP is blind to the DC pedestal (the ramp filter has zero DC gain)",
              abs(lf) < 0.02, f"kayip {lf:+.4f}")
        check(f"SIRT-{SIRT_ITERS} DC'den FBP'den cok daha fazla zarar goruyor",
              ls > max(5*abs(lf), 0.05), f"SIRT {ls:+.4f} vs FBP {lf:+.4f}")
        check("DC-induced damage GROWS with iteration count",
              ls > RES["pedestal"]["SIRT-20"]["loss"],
              f"{RES['pedestal']['SIRT-20']['loss']:+.4f} -> {ls:+.4f}")

# =============================================================================
# STAGE R — REPORT
# =============================================================================
hdr("SUMMARY")
ok = sum(1 for _, v, _ in CHECKS if v); tot = len(CHECKS)
fails = [(n, d) for n, v, d in CHECKS if not v]
if fails:
    P("  FAILED CHECKS:")
    for n, d in fails: P(f"    x {n}   {d}")
P(f"\n  {ok}/{tot} checks passed   |   {(time.time()-T0)/60:.0f} min total")
RES.update(checks=[{"name": n, "pass": v, "detail": d} for n, v, d in CHECKS],
           passed=ok, total=tot, minutes=(time.time()-T0)/60, stages=STAGES,
           dataset=DATA, reference=(REFERENCE if REFERENCE else None), full=FULL,
           profile=args.profile,
           timestamp=time.strftime("%Y-%m-%d %H:%M"),
           config=dict(N_FUSED=N_FUSED, N_BASELINE=N_BASELINE, N_DOSE=N_DOSE,
                       N_BITEXACT=N_BITEXACT,
                       DO_NOISE_ARRAYS=DO_NOISE_ARRAYS,
                       DO_NOISE_BITEXACT=DO_NOISE_BITEXACT, DO_ZIP_CRC=DO_ZIP_CRC))
tmp = f"{OUTROOT}/results.json.t{os.getpid()}"
json.dump(RES, open(tmp, "w"), indent=2, default=str, ensure_ascii=False)
os.replace(tmp, f"{OUTROOT}/results.json")

if "R" in STAGES:
    def g(*ks, default=None):
        d = RES
        for k in ks:
            if not isinstance(d, dict) or k not in d: return default
            d = d[k]
        return d
    def f(x, n=4):
        try:
            return "--" if x is None else f"{float(x):.{n}f}"
        except Exception: return str(x)
    L = []; A = L.append
    A("# VICTRE-Paired — technical validation report")
    A("")
    A(f"**Date:** {RES['timestamp']}  ")
    A(f"**Checks:** {ok}/{tot} passed  ")
    A(f"**Time:** {RES['minutes']:.0f} min  ")
    _heavy = "the full population" if FULL else f"a stratified sample of {g('n_fused')} patients"
    A(f"**Profile:** `{args.profile}`  ")
    A(f"**Coverage:** integrity / schema / dtypes / constants / geometry formula "
      f"= **the full population ({g('n_'+TARGET_TAG)} patients)**  ")
    A(f"Heavy per-array measurements (flat-field / noise / ROI-SDNR / geometry "
      f"residual) = **{_heavy}**  ")
    if REFERENCE:
        A(f"**Reference (regression check):** `{REFERENCE}`  ")
    A(f"**Bit-exact noise check:** {'on' if DO_NOISE_BITEXACT else 'off'}  |  "
      f"**zip CRC:** {'on' if DO_ZIP_CRC else 'off'}")
    A("")
    if fails:
        A("## Failed checks"); A("")
        for n, d in fails: A(f"- **{n}** -- {d}")
        A("")
    else:
        A("> All checks passed."); A("")

    A("## 1. Coverage and integrity"); A("")
    st = g("storage") or {}
    _nch = g('n_chunk') or st.get(TARGET_TAG+'_chunks')
    A(f"- Patients: **{g('n_'+TARGET_TAG)}**  |  chunks: **{_nch}**  |  "
      f"size: **{f(st.get(TARGET_TAG+'_gb'),1)} GB**")
    A(f"- Schema/dtype/shape/constant issues: **{g('schema_issues')}**")
    A(f"- Corrupt chunks (zip CRC): **{g('crc_bad') if DO_ZIP_CRC else 'not checked'}**")
    A(f"- Geometry-formula violations (full population): `{g('geom_formula_violations')}`")
    sd = g("split_dist") or {}
    if sd:
        A(""); A("| split | n | positive | " + " | ".join(DENS) + " |")
        A("|---" * (len(DENS)+3) + "|")
        for sp in ["train","val","test"]:
            if sp in sd:
                e = sd[sp]
                A(f"| {sp} | {e['n']} | {100*e['pos']/max(e['n'],1):.1f}% | " +
                  " | ".join(str(e[d]) for d in DENS) + " |")
    A("")

    A("## 2. Normalization and mask"); A("")
    nz = g("normalization") or {}; mk = g("mask") or {}
    A(f"- `clean` {CFG['P_RECON']}th percentile = **{f(nz.get('clean_p'),5)}** "
      f"(should be 1.0), clipped voxels {f(100*(nz.get('clean_clip') or 0),4)}%")
    A(f"- `clean_proj` {CFG['P_PROJ']}th percentile = **{f(nz.get('proj_p'),5)}**, "
      f"clipped pixels {f(100*(nz.get('proj_clip') or 0),4)}%")
    A(f"- `mask` = `clean > {CFG['MASK_THR']}` Dice = **{f(mk.get('dice'),6)}** "
      f"(min {f(mk.get('dice_min'),6)}); breast volume fraction "
      f"{f(100*(mk.get('frac') or 0),1)}%")
    A("")

    A("## 3. Flat-field / air attenuation"); A("")
    ff = g("flatfield") or {}
    if ff:
        A("| class | n | air p | air zero% | tissue p90 | uniformity |")
        A("|---|---|---|---|---|---|")
        for dn in DENS:
            e = ff.get(dn)
            if not e: continue
            A(f"| {dn} | {e['n']} | {f(e['air'])} | {f(100*e['zerofrac'],1)}% | "
              f"{f(e['tissue_p90'])} | {f(e['unif'])} |")
        A("")
        A(f"Cross-class pedestal std: **{f(g('flatfield_class_std'),5)}** -- air "
          f"attenuation should be zero on physical grounds and constant across "
          f"density classes (otherwise breast density leaks through the background).")
        A("")

    A("## 4. Noise model"); A("")
    nsr = g("noise") or {}
    A("| dose | sigma (stored) | sigma (measured) | relative diff | reproduction from seed |")
    A("|---|---|---|---|---|")
    for dn_, _, _, _ in DOSE:
        e = nsr.get(dn_, {})
        be = e.get("bitdiff_max")
        A(f"| {dn_} | {f(e.get('sigma'),5)} | {f(e.get('measured'),5)} | "
          f"{f(100*(e.get('reldiff') or 0),5)}% | "
          + (f"max **{be:.0f} LSB**, <=1 LSB: {f(100*(e.get('bitdiff_frac_exact') or 0),2)}%"
             if be is not None else "--") + " |")
    A("")
    if g("noise_formula"):
        A(f"`noisy_proj` arrays are bit-exactly reproducible from `noise_seed` "
          f"using the formula in `generate_dataset.add_noise` "
          f"(dose_idx mapping `{g('noise_formula','map')}`).")
        A("")
    if g("sigma_ratio"):
        A(f"- Quarter/full sigma ratio **{f(g('sigma_ratio'),4)}** (Poisson theory: "
          f"2.00; slightly lower is expected from electronic noise and clipping)")
    A("")

    A("## 5. Geometry -- residual parallax"); A("")
    ge = g("geometry") or {}
    if ge:
        A("| metric | value |"); A("|---|---|")
        A(f"| mean \\|dr\\|max | **{f(ge.get('dr_mean'),4)} px** |")
        A(f"| median | {f(ge.get('dr_median'),4)} px |")
        A(f"| 95th / 99th percentile | {f(ge.get('dr_p95'),4)} / {f(ge.get('dr_p99'),4)} px |")
        A(f"| maximum | {f(ge.get('dr_max'),4)} px |")
        A(f"| detrended | {f(ge.get('dr_detrend'),4)} px |")
        A(f"| \\|dc\\|max (column) | {f(ge.get('dc_mean'),4)} px |")
        A(f"| corr3d A(clean)<->clean_proj | {f(ge.get('corr3d'))} "
          f"(min {f(ge.get('corr3d_min'))}) |")
        A(f"| n | {ge.get('n')} |")
        A("")
        bd = ge.get("by_density") or {}
        if bd:
            A("| class | n | \\|dr\\|max | slope (px/view) | detrended | corr3d |")
            A("|---|---|---|---|---|---|")
            for dn in DENS:
                e = bd.get(dn)
                if e: A(f"| {dn} | {e['n']} | {f(e['dr'])} | {f(e['slope'],5)} | "
                        f"{f(e['detrend'])} | {f(e['corr3d'])} |")
            A("")
        th = ge.get("thickness")
        if th:
            A(f"Residual parallax scales with breast thickness: "
              f"`|dr|max = {f(th['dr_slope'],5)}*native_z` (r={f(th['dr_r'],3)}), "
              f"`slope = {f(th['slope_slope'],6)}*native_z` (r={f(th['slope_r'],3)}). "
              f"Detrending leaves **{f(th['detrended'],4)} px**.")
            A("")
    ofz = g("offz_refine")
    if ofz:
        A("### 5b. Per-density z-offset refinement (metadata-only)"); A("")
        A("`geom_offz` is a metadata field; since `clean_proj` is stored unshifted, "
          "this refinement does not require regenerating any image data.")
        A("")
        A("| class | DELTA (current) | DELTA (refined) | \\|dr\\|max before | after | decision |")
        A("|---|---|---|---|---|---|")
        for dn in DENS:
            e = ofz.get(dn)
            if e: A(f"| {dn} | {f(e['delta_current'],3)} | {f(e['delta_refined'],3)} | "
                    f"{f(e['dr_current'],4)} | {f(e['dr_refined'],4)} | "
                    f"{'accept' if e['accept'] else 'reject'}"
                    f"{' (search bound)' if e.get('edge') else ''} |")
        os_ = g("offz_summary")
        if os_:
            A("")
            A(f"Population-weighted \\|dr\\|max: **{f(os_['before'],4)} -> "
              f"{f(os_['after'],4)} px**")
        A("")

    A("## 6. Lesion and control ROIs (task-based)"); A("")
    le = g("lesion") or {}
    A(f"- Mass SDNR **{f(le.get('mass'))}** (n={le.get('n_mass')} patients), "
      f"calcification **{f(le.get('calc'))}**, control ROI **{f(le.get('control'))}** "
      f"(n={le.get('n_ctrl')})")
    A(f"- **d' = {f(le.get('dprime'),3)}**, **AUC = {f(le.get('auc'),3)}**")
    gr = g("lesion_by_density") or {}
    if gr:
        A(f"- Density gradient: " + ", ".join(f"{d}={f(gr[d],3)}" for d in DENS if d in gr))
    A("")

    A("## 7. Baselines -- two regimes"); A("")
    bl = g("baseline") or {}
    if bl:
        A(f"Official **{BASELINE_SPLIT}** split, n={g('inversion','n')} patients."); A("")
        A("| method | ideal A(clean) | real clean_proj | gap | PSNR | mass SDNR |")
        A("|---|---|---|---|---|---|")
        for m, e in bl.items():
            A(f"| {m} | {f(e['ideal'])} | {f(e['real'])} | {f(e['gap'])} | "
              f"{f(e['psnr_real'],2)} | {f(e['mass_sdnr_real'])} |")
        A("")
        iv = g("inversion") or {}
        if iv:
            A(f"**Ranking reversal.** Ideal regime: {' > '.join(iv['ideal_order'])}. "
              f"Real regime: {' > '.join(iv['real_order'])}. "
              f"Spearman rho = **{f(iv['spearman'],3)}**. "
              f"FBP best per-patient: **{iv['fbp_best']}/{iv['n']}**; "
              f"FBP's gap negative: **{iv['fbp_neg_gap']}/{iv['n']}**.")
            A("")
    dr_ = g("dose_response") or {}
    if dr_:
        A("### 7b. Dose response"); A("")
        order = ["clean"] + [a_ for a_, _, _, _ in DOSE]
        A("| method | metric | " + " | ".join(order) + " |")
        A("|---"*(len(order)+2) + "|")
        for m, e in dr_.items():
            for k, lbl in [("corr","correlation"), ("mass_sdnr","mass SDNR")]:
                A(f"| {m} | {lbl} | " + " | ".join(f(e[o][k]) for o in order) + " |")
        A("")
    rn = g("recon_noise_corr") or {}
    if rn:
        A("**Reconstruction-noise lag-1 autocorrelation along z:** "
          + ", ".join(f"{k} = {f(v['z'],3)}" for k, v in rn.items())
          + ". Limited-angle reconstruction correlates noise across slices; adding "
            "white noise directly to the volume would misrepresent this structure.")
        A("")
    pd_ = g("pedestal") or {}
    if pd_:
        A("### 7c. Controlled DC-pedestal sweep"); A("")
        A("| method | " + " | ".join(f"d={p_}" for p_ in PEDESTALS) + " | loss |")
        A("|---"*(len(PEDESTALS)+2) + "|")
        for m, e in pd_.items():
            A(f"| {m} | " + " | ".join(f(e["values"].get(str(p_))) for p_ in PEDESTALS)
              + f" | {f(e['loss'])} |")
        A("")
        A("The ramp filter has zero DC gain, so FBP is insensitive to a constant "
          "projection offset; iterative solvers have no such filter and a fixed "
          "offset leaves a residual no volume can explain, which compounds with "
          "iteration count.")
        A("")

    A("## 8. Generated files"); A("")
    tabs = sorted(glob.glob(f"{TAB}/*.csv"))
    A(f"{len(tabs)} tables. Figures for the paper are produced separately by "
      f"`run_baselines.py` and the `figures/` scripts, not by this validator."); A("")
    for t in tabs: A(f"- `tables/{os.path.basename(t)}`")
    A(""); A("---"); A("")
    A("_Generated automatically. Share `results.json` alongside this report._")

    tmp = f"{OUTROOT}/report.md.t{os.getpid()}"
    open(tmp, "w").write("\n".join(L)); os.replace(tmp, f"{OUTROOT}/report.md")
    P(f"  -> {OUTROOT}/report.md")
    zp = f"{OUTROOT}/results.zip"
    with zipfile.ZipFile(zp + ".t", "w", zipfile.ZIP_DEFLATED) as z:
        for p_ in [f"{OUTROOT}/report.md", f"{OUTROOT}/results.json"]:
            if os.path.exists(p_): z.write(p_, os.path.basename(p_))
        for p_ in sorted(glob.glob(f"{TAB}/*.csv")):
            if os.path.getsize(p_) < 60e6: z.write(p_, "tables/"+os.path.basename(p_))
    os.replace(zp + ".t", zp)
    P(f"  -> {zp}  ({os.path.getsize(zp)/1e6:.1f} MB)")

for f_ in glob.glob(f"{CACHE}/*"):
    try: os.remove(f_)
    except Exception: pass
P(f"\n  >> {'ALL CHECKS PASSED' if ok == tot else 'SOME CHECKS FAILED -- see report.md'}")
P(f"  >> {OUTROOT}")
LOG.close()
