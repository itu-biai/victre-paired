"""Generates 07_delta_patch_colab.ipynb. Recomputes sart_full, breast_mask and volume_box_mm of the dense/hetero/scattered
files with DELTA = 19.5; touches no other field."""
import json, sys
from pathlib import Path

OUT = Path(sys.argv[1])
cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md(r"""
# VICTRE-Paired — DELTA patch (Colab, GPU)

Notebook 06 showed that `DELTA` must be VICTRE's analytic value **19.5 mm** in all four densities
(the DELTA that zeroes the offset is 19.48 / 19.48 / 19.48 / 19.47). Fatty was already generated with 19.5;
dense / hetero / scattered with 19.765 / 20.242 / 20.091. This notebook patches the files of those three classes.

**What changes and what does not.** `DELTA` sets only the height of the volume grid.

| field | |
|---|---|
| `sart_full`, `breast_mask`, `volume_box_mm` | **recomputed** from the file's own `proj_full` + `proj_valid`, with the generator's own `Geometry.os_sart`, `breast_hull`, `breast_mask`, `volume_box` functions |
| projections and all low doses, DM, `victre_fbp`, `lesions`, `lesion_roi`, constants | copied **byte for byte** |

The one difference: the generator ran `os_sart` on a `float32` `proj_full`; the file stores `float16`.
The patched `sart_full` is built from the released `proj_full`, so it is **exactly reproducible from the data a
user has**. The check in B measures the size of the difference (rebuilt with the old DELTA and compared with the record).

**Safety.**
* Each file is first copied to the VM disk, patched there, written to Drive as `<id>.partial.npz` and renamed
  once its size is verified. If interrupted, the old file stays in place (every notebook skips
  `.partial` files).
* The file itself is the marker: if `volume_box_mm` shows the new DELTA, the file counts as patched and is
  skipped. Log: `delta_patch_s<SESSION>.jsonl`.
* For each file the tissue offset `sart_full ↔ victre_fbp` is measured: old ≈ −0.24 / −0.80 / −0.65, new ≈ 0.
  If a new |k| > 0.3 appears, the notebook **stops**.
* Every Drive operation has a timeout: if it hangs, the notebook stops and says what to do.

| | cell | when | time |
|---|---|---|---|
| **A** | setup | in every session | 2 min |
| **B** | settings + check | in every session; on the first run, one file per density rebuilt with the old DELTA = the record | 2 min |
| **C** | patch | resumes where it stopped; 2–3 tabs in parallel (`SESSION` / `N_SESSIONS`) | ~2,670 files × ~25 s ≈ 18 h in one tab |
| **D** | status | at any time | 1 min |
| **E** | finish | when all are done: size column of `manifest.csv` + `delta_patch.md` | 1 min |

**Then:** cell **F** of the generation notebook (`dataset_test.py`, `validation.md`), then 01 → 03 → 04 → 05 → 02.
""")

md("## A · Setup")
code(r"""
import subprocess, sys
out = subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
assert out, "NO GPU -> Runtime > Change runtime type > GPU"
print("GPU:", out)

!pip -q install astra-toolbox pydicom tqdm 2>&1 | tail -2
import astra
print("ASTRA", astra.__version__, "| CUDA:", astra.astra.use_cuda())
assert astra.astra.use_cuda(), "ASTRA has no CUDA build"

from google.colab import drive
drive.mount('/content/drive')
""")

md("## B · Settings + check")
code(r"""
import os, sys, json, time, shutil
from pathlib import Path
import numpy as np, pandas as pd
from scipy import ndimage
import concurrent.futures as _cf

NB_DIR = Path("/content/drive/MyDrive/victre_paired/code")
DATA   = Path("/content/drive/MyDrive/New_DBT/VICTRE-PAIRED-v10")
LOCAL  = Path("/content/patch")

SESSION, N_SESSIONS = 0, 1        # parallel tabs: 0..N-1, the same N in all
DELTA_OLD = {"dense": 19.765, "hetero": 20.242, "scattered": 20.091}   # before the patch
K_MAX_NEW = 0.30                  # STOP if the new tissue offset is larger than this
IO_TIMEOUT = 900                  # s, one Drive operation

sys.path.insert(0, str(NB_DIR)); os.chdir(NB_DIR)
import dataset_generate as G
assert all(v == 19.5 for v in G.DELTA.values()), \
    f"dataset_generate.py on Drive is old: {G.DELTA}. Upload the current dataset_generate.py to victre_paired/code/."
LOCAL.mkdir(parents=True, exist_ok=True)

man = pd.read_csv(DATA/"manifest.csv", dtype={"patient": str})
ITEMS = [(r.patient, r.density, r.split) for r in man.itertuples() if r.density in DELTA_OLD]
ITEMS.sort()
MINE = [it for i, it in enumerate(ITEMS) if i % N_SESSIONS == SESSION]
LOG = DATA/f"delta_patch_s{SESSION}.jsonl"
print(f"classes to patch: {sorted(DELTA_OLD)} | {len(ITEMS)} files in total, {len(MINE)} in this tab")

_IO = _cf.ThreadPoolExecutor(1); _PRE = _cf.ThreadPoolExecutor(1); _UP = _cf.ThreadPoolExecutor(1)
def timed(pool, fn, *a, what="Drive", timeout=IO_TIMEOUT):
    try:
        return pool.submit(fn, *a).result(timeout=timeout)
    except _cf.TimeoutError:
        raise RuntimeError(f"{what}: no response in {timeout} s. Runtime > Restart session, then A -> B -> C "
                           "(resumes where it stopped; an interrupted file stays as .partial, the old one is intact).")

def geo_d(dn, S, delta):
    old = G.DELTA[dn]; G.DELTA[dn] = float(delta)
    try: return G.Geometry(dn, S), np.array(G.volume_box(dn, S), np.float32)
    finally: G.DELTA[dn] = old

def highpass(v):
    v = v.astype(np.float32); return v - ndimage.gaussian_filter(v, (0, 4.0, 4.0))
def zshift(a, b, m, kmax=3, edge=2):
    # as in 06 C: the k (slices) that makes a[z + k] ~ b[z], inside the mask, parabolic peak
    S = a.shape[0]; ha, hb = highpass(a), highpass(b); w = m.any(0)
    ks = np.arange(-kmax, kmax + 1); c = np.full(len(ks), np.nan)
    for j, k in enumerate(ks):
        zs = [z for z in range(edge, S - edge) if 0 <= z + k < S]
        A = np.stack([ha[z + k][w] for z in zs]); B = np.stack([hb[z][w] for z in zs])
        A = A - A.mean(1, keepdims=True); B = B - B.mean(1, keepdims=True)
        c[j] = float((A * B).sum() / np.sqrt((A * A).sum() * (B * B).sum() + 1e-30))
    i = int(np.nanargmax(c))
    if i in (0, len(ks) - 1): return float(ks[i])
    y0, y1, y2 = c[i-1], c[i], c[i+1]; den = y0 - 2*y1 + y2
    return float(ks[i] + (np.clip(0.5 * (y0 - y2) / den, -1, 1) if abs(den) > 1e-12 else 0.0))

def recon(dn, S, delta, p32, valid):
    geo, box = geo_d(dn, S, delta)
    x = geo.os_sart(p32, valid)                                   # the SAME function as the generator
    mask = G.breast_mask(x, G.breast_hull(geo, p32, valid))       # the SAME functions as the generator
    return x.astype(np.float32), mask, box

def done_ids():
    out = {}
    for f in DATA.glob("delta_patch_s*.jsonl"):
        for l in f.read_text().splitlines():
            if l.strip():
                r = json.loads(l); out[r["patient"]] = r
    return out

# ---- check: one file per density; does a rebuild with the OLD DELTA give the recorded volume?
CHECK = DATA/"delta_patch_audit.json"
if not CHECK.exists():
    res = {}
    for dn in DELTA_OLD:
        pid, _, sp = next(it for it in ITEMS if it[1] == dn)
        src = DATA/sp/f"{pid}.npz"
        with np.load(timed(_IO, lambda: shutil.copy(src, LOCAL/"audit.npz"), what=src.name)) as z:
            d = {k: z[k] for k in ("sart_full", "proj_full", "proj_valid", "breast_mask", "volume_box_mm")}
        S = d["sart_full"].shape[0]
        assert np.allclose(d["volume_box_mm"], geo_d(dn, S, DELTA_OLD[dn])[1], atol=1e-3), \
            f"{pid}: volume_box_mm does not show the old DELTA -- already patched? The check needs an unpatched file."
        x, mk, _ = recon(dn, S, DELTA_OLD[dn], d["proj_full"].astype(np.float32), d["proj_valid"].astype(bool))
        ref = d["sart_full"]; hi = float(np.percentile(ref[d["breast_mask"]], 99.5))
        res[dn] = dict(patient=pid, max_abs=float(np.abs(x - ref).max()), max_rel_p995=float(np.abs(x - ref).max() / hi),
                       mean_abs_rel=float(np.abs(x - ref)[d["breast_mask"]].mean() / hi),
                       mask_agree=float((mk == d["breast_mask"]).mean()))
        print(f"{dn:<9} {pid}: rebuild with the old DELTA - record: max |diff| {res[dn]['max_abs']:.2e} "
              f"({100*res[dn]['max_rel_p995']:.2f} % of p99.5), mean {100*res[dn]['mean_abs_rel']:.3f} %; "
              f"mask agreement {res[dn]['mask_agree']:.5f}", flush=True)
        (LOCAL/"audit.npz").unlink()
    bad = [dn for dn, r in res.items() if r["max_rel_p995"] > 0.02 or r["mask_agree"] < 0.999]
    assert not bad, f"STOP: the rebuild does not give the recorded volume ({bad}). Is dataset_generate.py on Drive the one used for generation?"
    CHECK.write_text(json.dumps(res, indent=1))
    print("check OK: the recipe reproduces the record (including the float16 proj_full difference)")
else:
    print("check done earlier:", json.loads(CHECK.read_text()))
print(f"patched (all tabs): {len(done_ids())} / {len(ITEMS)}")
""")

md(r"""
## C · Patch

Resumes where it stopped. The next file is downloaded in the background while this one is processed; the upload
also runs in the background. Per line: download, rebuild, tissue offset **old → new**, mask ratio, upload.
""")
code(r"""
def fetch(it):
    pid, dn, sp = it
    dst = LOCAL/f"{pid}.in.npz"
    shutil.copy(DATA/sp/f"{pid}.npz", dst); return dst

def push(local, pid, sp, nbytes):
    part, tgt = DATA/sp/f"{pid}.partial.npz", DATA/sp/f"{pid}.npz"
    shutil.copy(local, part)
    assert part.stat().st_size == nbytes, f"{pid}: the size written to Drive does not match"
    os.replace(part, tgt)
    assert tgt.stat().st_size == nbytes, f"{pid}: the size does not match after renaming"
    local.unlink()

done = done_ids()
todo = [it for it in MINE if it[0] not in done]
print(f"this tab: {len(MINE)} files, {len(MINE)-len(todo)} finished, {len(todo)} left", flush=True)
t0, nxt, up = time.time(), None, None
for i, it in enumerate(todo, 1):
    pid, dn, sp = it
    ts = time.time()
    fut = nxt if nxt is not None else _PRE.submit(fetch, it)
    try:
        src = fut.result(timeout=IO_TIMEOUT)
    except _cf.TimeoutError:
        raise RuntimeError(f"{pid}: download, no response in {IO_TIMEOUT} s. Restart session, A -> B -> C.")
    nxt = _PRE.submit(fetch, todo[i]) if i < len(todo) else None
    t_dl = time.time() - ts
    with np.load(src) as z:
        d = {k: z[k] for k in z.files}
    S = d["sart_full"].shape[0]
    box_new = geo_d(dn, S, 19.5)[1]
    if np.allclose(d["volume_box_mm"], box_new, atol=1e-3):
        rec = dict(patient=pid, density=dn, split=sp, status="already_patched")
        with open(LOG, "a") as h: h.write(json.dumps(rec) + "\n")
        src.unlink(); print(f"[{i}/{len(todo)}] {pid} already patched, skipped", flush=True); continue
    assert np.allclose(d["volume_box_mm"], geo_d(dn, S, DELTA_OLD[dn])[1], atol=1e-3), \
        f"STOP: {pid} volume_box_mm is neither the old nor the new DELTA: {d['volume_box_mm']}"
    tr = time.time()
    x, mk, box = recon(dn, S, 19.5, d["proj_full"].astype(np.float32), d["proj_valid"].astype(bool))
    t_rc = time.time() - tr
    rec = dict(patient=pid, density=dn, split=sp, status="patched", S=S,
               mask_ratio=float(mk.sum() / max(d["breast_mask"].sum(), 1)))
    if bool(d["victre_fbp_present"]):
        vf = d["victre_fbp"].astype(np.float32) / 65535.0 * float(d["victre_fbp_scale"])
        rec["k_old"] = zshift(d["sart_full"], vf, d["breast_mask"]); rec["k_new"] = zshift(x, vf, mk)
        if abs(rec["k_new"]) > K_MAX_NEW:
            raise RuntimeError(f"STOP: {pid} new tissue offset {rec['k_new']:+.2f} slices (> {K_MAX_NEW}). "
                               "The file was NOT CHANGED. Report this result.")
    shapes = {k: (v.shape, v.dtype) for k, v in d.items()}
    d["sart_full"], d["breast_mask"], d["volume_box_mm"] = x, mk, box
    assert {k: (v.shape, v.dtype) for k, v in d.items()} == shapes, f"{pid}: shape/type changed"
    outp = LOCAL/f"{pid}.partial.npz"
    np.savez_compressed(outp, **d); del d
    src.unlink()
    nbytes = outp.stat().st_size; rec["size_mb"] = round(nbytes / 1e6, 1)
    if up is not None:
        try: up.result(timeout=IO_TIMEOUT)
        except _cf.TimeoutError:
            raise RuntimeError(f"upload, no response in {IO_TIMEOUT} s. Restart session, A -> B -> C.")
    loc = LOCAL/f"{pid}.out.npz"; outp.rename(loc)
    up = _UP.submit(push, loc, pid, sp, nbytes)
    up.result(timeout=IO_TIMEOUT) if i == len(todo) else None
    # the log is written when the upload has finished: an interrupted upload does not count as finished
    def _log(f_, rec=rec):
        if f_.exception() is None:
            with open(LOG, "a") as h: h.write(json.dumps(rec) + "\n")
    up.add_done_callback(_log)
    el = time.time() - t0
    ks = f"k {rec['k_old']:+.2f} -> {rec['k_new']:+.2f}" if "k_new" in rec else "k (no victre_fbp)"
    print(f"[{i}/{len(todo)}] {pid} {dn:<9} download {t_dl:.0f}s rebuild {t_rc:.0f}s | {ks} | mask x{rec['mask_ratio']:.3f} "
          f"| ~{el/i*(len(todo)-i)/3600:.1f} h left", flush=True)
if up is not None: up.result(timeout=IO_TIMEOUT)
time.sleep(1)
print(f"\nthis tab is done. All tabs: {len(done_ids())} / {len(ITEMS)}")
""")

md("## D · Status")
code(r"""
D = pd.DataFrame(done_ids().values())
print(f"patched: {len(D)} / {len(ITEMS)}")
if len(D):
    print(D.groupby(["density", "status"]).size().unstack(fill_value=0))
    if "k_new" in D:
        print("\ntissue offset (slices), old -> new:")
        print(D.groupby("density")[["k_old", "k_new"]].agg(["mean", "std", "min", "max"]).round(3).to_string())
    print("\nmask voxel ratio new/old:", D.groupby("density").mask_ratio.describe()[["mean", "min", "max"]].round(4).to_dict("index"))
""")

md(r"""
## E · Finish — when all are done

Every file's `volume_box_mm` must show the new DELTA (cell C2 of 01 checks this independently too).
Only `size_mb` is updated in `manifest.csv` (no other column depends on DELTA). Record: `delta_patch.md`.
""")
code(r"""
D = pd.DataFrame(done_ids().values())
miss = sorted(set(p for p, _, _ in ITEMS) - set(D.patient))
assert not miss, f"{len(miss)} files not patched: {miss[:5]} -> complete C"
leftover = [p for s in ("train", "val", "test") for p in (DATA/s).glob("*.partial.npz")]
assert not leftover, f"interrupted .partial files: {leftover[:3]} -> run C again (it overwrites them)"
sz = dict(zip(D.patient, D.get("size_mb", pd.Series(dtype=float))))
man2 = man.copy()
man2["size_mb"] = [sz[p] if p in sz and np.isfinite(sz.get(p, np.nan)) else v for p, v in zip(man2.patient, man2.size_mb)]
man2.to_csv(DATA/"manifest.csv", index=False)
Y = D[D.status == "patched"]
lines = ["# DELTA patch", "",
         f"Date: {time.strftime('%Y-%m-%d %H:%M')}. DELTA dense/hetero/scattered {DELTA_OLD} -> 19.5 mm (fatty already 19.5).",
         "Recomputed fields: sart_full, breast_mask, volume_box_mm (from proj_full + proj_valid, with the generator's own",
         "functions). All other fields were copied unchanged.", "",
         f"Patched files: {len(Y)} (+ {int((D.status == 'already_patched').sum())} already patched).", "",
         "| density | n | tissue offset old (slices) | new | mask ratio new/old |", "|---|---|---|---|---|"]
for dn, q in Y.groupby("density"):
    ko = f"{q.k_old.mean():+.3f} ± {q.k_old.std():.3f}" if "k_old" in q and q.k_old.notna().any() else "-"
    kn = f"{q.k_new.mean():+.3f} ± {q.k_new.std():.3f}" if "k_new" in q and q.k_new.notna().any() else "-"
    lines.append(f"| {dn} | {len(q)} | {ko} | {kn} | {q.mask_ratio.mean():.4f} [{q.mask_ratio.min():.4f}, {q.mask_ratio.max():.4f}] |")
lines += ["", "Check (rebuild with the old DELTA = record):", "```", CHECK.read_text(), "```"]
(DATA/"delta_patch.md").write_text("\n".join(lines))
print("\n".join(lines))
print("\nNEXT: generation notebook F (dataset_test.py -> validation.md), then 01 -> 03 -> 04 -> 05 -> 02.")
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
