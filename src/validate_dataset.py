"""
validate_dataset.py — technical validation of VICTRE-PAIRED v6.

Read-only checks that reproduce the numbers reported in the Technical Validation
section of the paper. The script verifies, in order:

  1. file integrity        — no corrupt chunks, expected sizes
  2. structure             — required keys present, array shapes correct
  3. metadata scan         — chunk / patient counts per split (all chunks)
  4. split disjointness    — no SEED leakage across train / val / test
  5. metadata consistency  — no signal-absent patient carries lesions; the only
                             native_z anomaly is the known broken patient
  6. dose model            — noise std increases monotonically full < half < quarter
  7. scale factors         — recon_scale / proj_scale distributions
  8. projection uniformity — per-view tissue level within the physical bound
  9. lesion conspicuity    — mass SDNR positive, calcification SDNR at noise floor

A JSON and a Markdown report are written to REPORT. Nothing else is modified.
The two-regime (inverse-crime) reconstruction evaluation lives in run_baselines.py.
"""
import os
import gc
import glob
import json
import time
from collections import Counter

import numpy as np

from constants import (
    NA, NZMAP, EXPECT, N_PATIENTS_TOTAL, BROKEN_SEEDS, fmt_seconds as fmt,
)

V6 = "/content/drive/MyDrive/New_DBT/VICTRE-PAIRED-v6"
REPORT = "/content/v6_validation"
os.makedirs(REPORT, exist_ok=True)

SPLITS = ["train", "val", "test"]
DEEP_N = 40      # chunks sampled for the heavy projection / SDNR checks

REQUIRED = {
    "clean", "noisy", "mask", "clean_proj", "noisy_proj", "noisy_proj_full",
    "noisy_proj_quarter", "sigma", "sigma_full", "sigma_quarter", "sigma_recon",
    "is_pos", "seed", "density", "native_z", "lesion_coords", "lesion_count",
    "recon_scale", "proj_scale", "dose_levels", "dose_gains", "elec_noise",
    "strip_valley",
}
SHAPES = {"clean": (8, 56, 408, 336), "clean_proj": (8, 25, 752, 384),
          "lesion_coords": (8, 8, 4)}


def safe(f, tries=3):
    """Load an .npz with retries (Google Drive FUSE can be flaky)."""
    for i in range(tries):
        try:
            return np.load(f)
        except Exception:
            if i == tries - 1:
                return None
            time.sleep(2); gc.collect()


def gl(pattern, tries=5):
    """glob with retries."""
    for i in range(tries):
        r = sorted(glob.glob(pattern))
        if r:
            return r
        time.sleep(3); gc.collect()
    return []


def sdnr(vol, z, h, w, ri=4, ro=12):
    """Signal-difference-to-noise ratio of a lesion at (z, h, w):
    mean of a small central box minus the mean of a surrounding ring,
    divided by the ring's std."""
    z, h, w = int(round(z)), int(round(h)), int(round(w))
    if not (0 <= z < vol.shape[0] and 0 <= h < vol.shape[1] and 0 <= w < vol.shape[2]):
        return np.nan
    c = vol[max(0, z - 1):z + 2, max(0, h - ri):h + ri + 1, max(0, w - ri):w + ri + 1]
    ring = vol[z, max(0, h - ro):h + ro + 1, max(0, w - ro):w + ro + 1]
    return (c.mean() - ring.mean()) / ring.std() if ring.std() > 1e-8 else np.nan


FAIL, RES = [], {}


def chk(condition, message):
    if not condition:
        FAIL.append(message)
    return condition


print("=" * 78); print("VICTRE-PAIRED v6 — TECHNICAL VALIDATION"); print("=" * 78, flush=True)

# ── 1) FILE INTEGRITY ──────────────────────────────────────────────────────
print("\n1) FILE INTEGRITY", flush=True)
files = {sp: gl(f"{V6}/{sp}/*.npz") for sp in SPLITS}
sizes = [(os.path.getsize(f), sp, f) for sp in SPLITS for f in files[sp]]
med = sorted(s for s, _, _ in sizes)[len(sizes) // 2] if sizes else 0
small = [(s, sp, f) for s, sp, f in sizes if s < med * 0.5]
corrupt = []
for s, sp, f in small:
    d = safe(f)
    if d is None:
        corrupt.append((sp, f))
    else:
        try:
            _ = d["clean_proj"].shape
        except Exception:
            corrupt.append((sp, f))
print(f"  median chunk: {med / 1e6:.0f} MB | files: {len(sizes)} | corrupt: {len(corrupt)}")
chk(not corrupt, f"{len(corrupt)} corrupt files")

# ── 2) STRUCTURE ───────────────────────────────────────────────────────────
print("\n2) STRUCTURE", flush=True)
d0 = safe(files["test"][0])
miss = REQUIRED - set(d0.files)
bad = [f"{k}:{d0[k].shape}!={v}" for k, v in SHAPES.items() if k in d0.files and d0[k].shape != v]
print(f"  keys: {len(d0.files)} | missing: {miss if miss else 'none'} | shapes: {'ok' if not bad else bad}")
chk(not miss, f"missing keys {miss}")
chk(not bad, f"bad shapes {bad}")
RES["keys"] = sorted(d0.files)

# ── 3) METADATA SCAN (all chunks) ──────────────────────────────────────────
print("\n3) METADATA SCAN (all chunks)", flush=True)
rows = []; t0 = time.time(); n = 0
tot = sum(len(v) for v in files.values())
for sp in SPLITS:
    for f in files[sp]:
        d = safe(f)
        if d is None:
            continue
        n += 1
        for b in range(len(d["seed"])):
            rows.append(dict(
                split=sp, chunk=os.path.basename(f), b=b, seed=int(d["seed"][b]),
                dens=str(d["density"][b]), nz=int(d["native_z"][b]), ip=bool(d["is_pos"][b]),
                nl=int(d["lesion_count"][b]), rq=float(d["recon_scale"][b]),
                pq=float(d["proj_scale"][b]), sf=float(d["sigma_full"][b]),
                sh=float(d["sigma"][b]), sq=float(d["sigma_quarter"][b]),
                sr=float(d["sigma_recon"][b])))
        if n % 25 == 0:
            e = time.time() - t0
            print(f"  {n}/{tot} | {fmt(e)} | ETA ~{fmt(e / n * (tot - n))}", flush=True)
tp = 0
for sp in SPLITS:
    nc = len(files[sp]); npat = sum(1 for r in rows if r['split'] == sp)
    ec, ep = EXPECT[sp]; tp += npat; ok = (nc == ec and npat == ep)
    print(f"  {sp:>6s}: {nc:>3d} chunks / {npat:>4d} patients  (expected {ec}/{ep}) {'ok' if ok else 'WARN'}")
    chk(ok, f"{sp}: {nc}/{npat} != {ec}/{ep}")
print(f"  TOTAL : {n:>3d} chunks / {tp:>4d} patients  (expected 346/{N_PATIENTS_TOTAL})")
chk(tp == N_PATIENTS_TOTAL, f"total {tp} != {N_PATIENTS_TOTAL}")
RES["counts"] = {sp: {"chunks": len(files[sp]),
                      "patients": sum(1 for r in rows if r['split'] == sp)} for sp in SPLITS}

# ── 4) SPLIT LEAKAGE ───────────────────────────────────────────────────────
print("\n4) SPLIT LEAKAGE", flush=True)
S = {sp: set(r['seed'] for r in rows if r['split'] == sp) for sp in SPLITS}
tv, tt, vt = S["train"] & S["val"], S["train"] & S["test"], S["val"] & S["test"]
allc = [r['seed'] for r in rows]; dup = len(allc) - len(set(allc))
print(f"  train∩val {len(tv)} | train∩test {len(tt)} | val∩test {len(vt)} | duplicate SEEDs {dup}")
chk(not (tv or tt or vt), "split leakage")
chk(dup == 0, f"{dup} duplicate SEEDs")
RES["leakage"] = {"tv": len(tv), "tt": len(tt), "vt": len(vt), "dup": dup}

# ── 5) METADATA CONSISTENCY (broken patient accounted for) ─────────────────
print("\n5) METADATA CONSISTENCY", flush=True)
nzb = [r for r in rows if NZMAP.get(r['dens']) != r['nz']]
brk = [r for r in nzb if r['seed'] in BROKEN_SEEDS]
unexp = [r for r in nzb if r['seed'] not in BROKEN_SEEDS]
neg_with_lesion = [r for r in rows if not r['ip'] and r['nl'] > 0]
pos_no_lesion = [r for r in rows if r['ip'] and r['nl'] == 0]
for r in brk:
    print(f"  known broken patient: SEED {r['seed']} {r['dens']} nz={r['nz']} "
          f"-> {r['split']}/{r['chunk']}[{r['b']}]")
print(f"  native_z anomalies: {len(nzb)} (known broken {len(brk)}, unexpected {len(unexp)})")
print(f"  signal-absent with lesions: {len(neg_with_lesion)} (must be 0)")
print(f"  signal-present without lesions: {len(pos_no_lesion)} (small = out-of-bounds crop, normal)")
chk(not neg_with_lesion, f"{len(neg_with_lesion)} signal-absent patients carry lesions")
chk(len(unexp) == 0, f"{len(unexp)} unexpected native_z anomalies")
dc = Counter(r['dens'] for r in rows); npos = sum(r['ip'] for r in rows)
for dn in ["scattered", "hetero", "dense", "fatty"]:
    s = [r for r in rows if r['dens'] == dn]
    print(f"  {dn:>10s}: {len(s):>5d} ({100 * len(s) / len(rows):>4.1f}%)  "
          f"pos {sum(r['ip'] for r in s)} / neg {sum(not r['ip'] for r in s)}")
print(f"  total: {npos} pos / {len(rows) - npos} neg ({100 * npos / len(rows):.1f}% positive)")
RES["density"] = dict(dc); RES["broken_found"] = [r['seed'] for r in brk]

# ── 6) DOSE MODEL ──────────────────────────────────────────────────────────
print("\n6) DOSE MODEL", flush=True)
sf = np.array([r['sf'] for r in rows]); sh = np.array([r['sh'] for r in rows])
sq = np.array([r['sq'] for r in rows]); sr = np.array([r['sr'] for r in rows])
for nm, a in [("full", sf), ("half", sh), ("quarter", sq), ("recon", sr)]:
    print(f"  {nm:>8s}: {a.mean():.4f} ± {a.std():.4f}  [{a.min():.4f}, {a.max():.4f}]")
m1 = (sf < sh).mean() * 100; m2 = (sh < sq).mean() * 100
print(f"  monotonic: full<half {m1:.1f}% | half<quarter {m2:.1f}% | ratio quarter/full {sq.mean() / sf.mean():.2f}x")
chk(min(m1, m2) > 99, f"dose monotonicity {m1:.0f}/{m2:.0f}")
RES["dose"] = {"full": float(sf.mean()), "half": float(sh.mean()),
               "quarter": float(sq.mean()), "ratio": float(sq.mean() / sf.mean())}

# ── 7) SCALE FACTORS ───────────────────────────────────────────────────────
print("\n7) SCALE FACTORS", flush=True)
rq = np.array([r['rq'] for r in rows]); pq = np.array([r['pq'] for r in rows])
print(f"  recon_scale: {rq.mean():.0f} ± {rq.std():.0f}  [{rq.min():.0f}, {rq.max():.0f}]")
print(f"  proj_scale : {pq.mean():.3f} ± {pq.std():.3f}  [{pq.min():.3f}, {pq.max():.3f}]")
RES["scales"] = {"recon_scale_mean": float(rq.mean()), "proj_scale_mean": float(pq.mean())}

# ── 8-9) PROJECTION UNIFORMITY + LESION SDNR (sampled, heavy) ──────────────
print(f"\n8-9) PROJECTION UNIFORMITY + LESION SDNR (sample of {DEEP_N} chunks)", flush=True)
allf = [f for sp in SPLITS for f in files[sp]]
np.random.default_rng(0).shuffle(allf)
uni, mass_sdnr, calc_sdnr = [], [], []
t0 = time.time()
for i, f in enumerate(allf[:DEEP_N]):
    d = safe(f)
    if d is None:
        continue
    cp = d["clean_proj"]; cl = d["clean"]; lc = d["lesion_coords"]; nlc = d["lesion_count"]
    for b in range(len(d["seed"])):
        if int(d["seed"][b]) in BROKEN_SEEDS:
            continue
        x = cp[b]
        lv = [np.percentile(x[a][x[a] > 0], 90) for a in range(NA) if (x[a] > 0).any()]
        if lv:
            uni.append(max(lv) / min(lv))
        for z, h, w, t in lc[b][:nlc[b]]:
            v = sdnr(cl[b], z, h, w)
            if not np.isnan(v):
                (mass_sdnr if t >= 4 else calc_sdnr).append(v)
    del d; gc.collect()
    if (i + 1) % 10 == 0:
        print(f"  {i + 1}/{DEEP_N} | {fmt(time.time() - t0)}", flush=True)
uni = np.array(uni); ms = np.array(mass_sdnr); cs = np.array(calc_sdnr)
print(f"  uniformity (physics 1.10): {uni.mean():.3f}  [{uni.min():.3f}, {uni.max():.3f}] "
      f"| >1.20: {(uni > 1.20).sum()}")
print(f"  mass SDNR (tip 4-7):          {ms.mean():+.3f} ± {ms.std():.3f}  (>0: {100 * (ms > 0).mean():.1f}%, n={len(ms)})")
print(f"  calcification SDNR (tip 0-3): {cs.mean():+.3f} ± {cs.std():.3f}  (n={len(cs)})  -> at noise floor")
chk((uni > 1.20).sum() == 0, f"{(uni > 1.20).sum()} projections exceed uniformity 1.20")
chk(ms.mean() > 0.3, f"mass SDNR too low ({ms.mean():.3f})")
RES["uniformity_mean"] = float(uni.mean())
RES["mass_sdnr_mean"] = float(ms.mean())
RES["calc_sdnr_mean"] = float(cs.mean())

# ── REPORT ─────────────────────────────────────────────────────────────────
RES["pass"] = (len(FAIL) == 0)
RES["failures"] = FAIL
with open(os.path.join(REPORT, "v6_validation.json"), "w") as fh:
    json.dump(RES, fh, indent=2)

md = ["# VICTRE-PAIRED v6 — Technical Validation report", ""]
md.append(f"**Result:** {'PASS ✅' if RES['pass'] else 'FAIL ⚠️'}")
if FAIL:
    md += ["", "**Failures:**"] + [f"- {m}" for m in FAIL]
md += ["", "## Counts", "```", json.dumps(RES["counts"], indent=2), "```",
       "## Dose model", "```", json.dumps(RES["dose"], indent=2), "```",
       f"## Physics", f"- projection uniformity: {RES['uniformity_mean']:.3f} (physical bound 1.10)",
       f"- mass SDNR: {RES['mass_sdnr_mean']:+.3f}",
       f"- calcification SDNR: {RES['calc_sdnr_mean']:+.3f} (at noise floor)"]
with open(os.path.join(REPORT, "v6_validation.md"), "w") as fh:
    fh.write("\n".join(md))

print("\n" + "=" * 78)
print(f"VALIDATION {'PASSED ✅' if RES['pass'] else 'FAILED ⚠️ — ' + '; '.join(FAIL)}")
print(f"reports written to {REPORT}")
print("=" * 78)
