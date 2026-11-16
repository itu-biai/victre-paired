"""Generates 04_training_colab.ipynb. One small experiment showing that paired training works (supplementary material)."""
import json, sys
from pathlib import Path

OUT = Path(sys.argv[1])
cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md(r"""
# VICTRE-Paired v11 — training experiment (Colab, GPU, supplementary material)

**The aim is not to compare models:** it is to show that paired training runs end to end on these data and
that the training split is usable. Supplementary training figure and table of the Data Descriptor.

**Experiment:** a single 2D U-Net (residual learning, slice by slice) that maps the low-dose OS-SART reconstruction
to `sart_full`. It is trained at the `TRAIN_DOSES` dose and tested at 50 / 25 / 10 %. The default is
25 % only; with `TRAIN_DOSES = ["50", "25", "10"]` the training × test heat map is produced
(~3 times the time).

| | cell | what it does | time (25 % only) |
|---|---|---|---|
| **A** | setup | GPU, ASTRA, Drive | 2 min |
| **B** | settings | paths, selection, U-Net | 1 min |
| **C** | data preparation | low-dose OS-SART of the training/validation phantoms + target slices → cache on Drive | ~1–2 h (depends on the Drive read speed) |
| **D** | training | U-Net, validation curve, checkpoint on Drive | ~1 h |
| **E** | test | test split, 3 doses × MC + 25 % synthetic; the input OS-SART on the same phantoms | ~1.5 h |
| **E2** | example slice | if `example.npz` is missing, produces it for the first test phantom with a mass | 1 min |
| **F** | collect | CSV files | 1 min |

**Every cell resumes where it stopped** (cache, checkpoint, one JSON per phantom).
If the session drops: **A → B →** the cell where you stopped. **First a trial:** `TRIAL = True` (few phantoms, few steps).

**Scope:** reporting only. The place of the learned method in the ranking is not interpreted in the Data Descriptor.

Outputs: `/content/drive/MyDrive/victre_paired/analysis/training/`
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
assert astra.astra.use_cuda(), "ASTRA has no CUDA build"

from google.colab import drive
drive.mount('/content/drive')
""")

md("## B · Settings")
code(r"""
import os, sys, json, time, math
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy import ndimage

NB_DIR = Path("/content/drive/MyDrive/victre_paired/code")
DATA   = Path("/content/drive/MyDrive/victre_paired/data")
RES    = Path("/content/drive/MyDrive/victre_paired/analysis")
EG     = RES/"training"

TRIAL          = False            # True: 2 phantoms/density, 300 steps, 3 test phantoms
TRAIN_DOSES     = ["25"]           # ["50","25","10"] -> training x test heat map
N_TRAIN_PER_DEN = 2 if TRIAL else 150     # training phantoms / density (4 x 150 = 600)
N_VAL_PER_DEN   = 1 if TRIAL else 10
SLICES_PER_PAT  = 16               # training slices per phantom (those with the largest breast area)
STEPS           = 300 if TRIAL else 30_000
BATCH, CROP, LR = 8, 256, 1e-3
VAL_EVERY       = 100 if TRIAL else 1_000
N_TEST          = 3 if TRIAL else 0       # 0: the whole test split

for d in (EG, EG/"cache", EG/"ckpt", EG/"test"):
    d.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(NB_DIR)); os.chdir(NB_DIR)
import dataset_generate as G
import dataset_test as T
DENS, DOSES = G.DENSITIES, T.DOSES
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)

files = {s: sorted(p for p in (DATA/s).glob("*.npz") if ".partial" not in p.name) for s in ("train", "val", "test")}
print({s: len(v) for s, v in files.items()})

def by_density(fs, n_per):
    # If a manifest exists, the density is read from it (fast); otherwise from the file. Deterministic: name order.
    # The generator's manifest.csv (cell G) or the registry.csv of 01: density without opening the files.
    import pandas as pd
    dmap = {}
    for kp in (DATA/"manifest.csv", RES/"registry.csv"):
        if kp.exists():
            K = pd.read_csv(kp, dtype={"patient": str}); dmap = dict(zip(K.patient, K.density)); break
    if not dmap: print("WARNING: no manifest.csv / registry.csv -> densities will be read from the files (slow)", flush=True)
    out = {dn: [] for dn in DENS}
    for f in fs:
        dn = dmap.get(f.stem)
        if dn is None:
            with np.load(f) as z: dn = str(z["density"])
        if len(out[dn]) < n_per: out[dn].append(f)
        if all(len(v) >= n_per for v in out.values()): break
    return [f for dn in DENS for f in out[dn]]
TRAIN = by_density(files["train"], N_TRAIN_PER_DEN)
VALP  = by_density(files["val"], N_VAL_PER_DEN)
TEST  = files["test"][:N_TEST] if N_TEST else files["test"]
print(f"training {len(TRAIN)}, validation {len(VALP)}, test {len(TEST)} phantoms | training doses {TRAIN_DOSES}")

# ------------------------------------------------------------------ U-Net (residual learning)
def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(),
                         nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU())
class UNet(nn.Module):
    def __init__(self, ch=32, depth=4):
        super().__init__()
        cs = [ch * 2**i for i in range(depth)]
        self.down = nn.ModuleList([block(1 if i == 0 else cs[i-1], cs[i]) for i in range(depth)])
        self.mid = block(cs[-1], cs[-1] * 2)
        self.up = nn.ModuleList([nn.ConvTranspose2d((cs[i] * 2), cs[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([block(cs[i] * 2, cs[i]) for i in reversed(range(depth))])
        self.out = nn.Conv2d(cs[0], 1, 1)
    def forward(self, x):
        skips, h = [], x
        for b in self.down:
            h = b(h); skips.append(h); h = F.max_pool2d(h, 2)
        h = self.mid(h)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            h = dec(torch.cat([up(h), s], 1))
        return x + self.out(h)

def pad16(x):              # (..., H, W) -> a multiple of 16
    H, W = x.shape[-2:]; ph, pw = (-H) % 16, (-W) % 16
    return F.pad(x, (0, pw, 0, ph), mode="replicate"), (H, W)

@torch.no_grad()
def apply_net(net, vol, scale):
    # Volume (S, H, W) slice by slice. Input and output in mm^-1; the network works on scaled values.
    net.eval(); out = []
    for i in range(0, vol.shape[0], 8):
        x = torch.from_numpy(np.ascontiguousarray(vol[i:i+8], np.float32))[:, None].to(DEV) / scale
        xp, (H, W) = pad16(x)
        with torch.autocast(DEV, enabled=(DEV == "cuda")):
            y = net(xp)
        out.append((y[..., :H, :W].float() * scale)[:, 0].cpu().numpy())
    return np.concatenate(out).astype(np.float32)

# the same synthetic dose model as in 03 (the syn column of the "learned" row of Table 4)
def synth_dose(p, footprint, n0, sigma_e, f, rng):
    counts = f * n0 * np.exp(-p.astype(np.float64))
    meas = counts + rng.standard_normal(p.shape) * np.sqrt(counts + sigma_e**2)
    return np.where(footprint, np.log(f * n0 / np.clip(meas, G.CFG.min_counts, None)), 0.0).astype(np.float32)

R_CORE, R_BG = 4, 12
def sdnr_at(vol, z, h, w):
    z, h, w = int(round(z)), int(round(h)), int(round(w))
    if not (0 <= z < vol.shape[0] and 0 <= h < vol.shape[1] and 0 <= w < vol.shape[2]): return None
    core = vol[max(0, z-1):z+2, max(0, h-R_CORE):h+R_CORE+1, max(0, w-R_CORE):w+R_CORE+1]
    bg   = vol[z, max(0, h-R_BG):h+R_BG+1, max(0, w-R_BG):w+R_BG+1]
    s = float(bg.std()); return None if s < 1e-12 else float((core.mean() - bg.mean()) / s)

def corr_masked(a, b, m):
    x = a[m].astype(np.float64); y = b[m].astype(np.float64); x -= x.mean(); y -= y.mean()
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
print("ready | device", DEV)
""")

md(r"""
## C · Data preparation — cache on Drive, resumes where it stopped

For each training and validation phantom, at each training dose: low-dose OS-SART (the dataset's
recipe, `Geometry.os_sart`), the target `sart_full` and the mask. `SLICES_PER_PAT` evenly spaced slices are chosen
among those with the largest breast area (for validation the whole volume is kept).
""")
code(r"""
t0 = time.time(); made = 0
for split, fs in (("train", TRAIN), ("val", VALP)):
    for j, f in enumerate(fs):
        for k in TRAIN_DOSES:
            out = EG/"cache"/f"{split}_{k}_{f.stem}.npz"
            if out.exists(): continue
            with np.load(f) as z:
                gt, m, valid = z["sart_full"], z["breast_mask"], z["proj_valid"]
                p = z[f"proj_{k}"].astype(np.float32); dn = str(z["density"])
            x = G.Geometry(dn, gt.shape[0]).os_sart(p, valid)
            if split == "train":
                area = m.reshape(m.shape[0], -1).sum(1)
                cand = np.where(area >= 0.5 * area.max())[0]
                sl = cand[np.linspace(0, len(cand) - 1, min(SLICES_PER_PAT, len(cand))).round().astype(int)]
            else:
                sl = np.arange(gt.shape[0])
            np.savez(out, inp=x[sl].astype(np.float16), tgt=gt[sl].astype(np.float16), mask=m[sl], density=dn)
            made += 1
        el = time.time() - t0
        print(f"  {split}: {j+1}/{len(fs)}  {f.stem}  | {el/60:.1f} min, {made} new files"
              + (f", {el/made:.0f} s/file" if made else ""), flush=True)
print(f"cache: {made} new files, {(time.time()-t0)/60:.1f} min")
""")

md(r"""
## D · Training

Masked MSE (`‖f_θ(x) − x_ref‖²` inside the breast), AdamW, cosine learning rate, random 256×256
crops, horizontal/vertical flips. Every `VAL_EVERY` steps, PSNR on the validation phantoms
(`dataset_test.metrics`, on the volume) → `training/curve_<dose>.csv`. Checkpoint on Drive.
""")
code(r"""
import pandas as pd
def load_cache(split, k):
    X, Y, Mk = [], [], []
    for f in sorted((EG/"cache").glob(f"{split}_{k}_*.npz")):
        with np.load(f) as z: X.append(z["inp"]); Y.append(z["tgt"]); Mk.append(z["mask"])
    return X, Y, Mk

for k in TRAIN_DOSES:
    ck = EG/"ckpt"/f"unet_{k}.pt"
    Xtr, Ytr, Mtr = load_cache("train", k)
    Xtr, Ytr, Mtr = np.concatenate(Xtr), np.concatenate(Ytr), np.concatenate(Mtr)
    Xva, Yva, Mva = load_cache("val", k)
    SCALE = float(np.percentile(Ytr[Mtr].astype(np.float32), 99))      # network input to ~[0,1]
    print(f"dose {k}%: {len(Xtr)} training slices, {len(Xva)} validation volumes, scale {SCALE:.4f}")

    net = UNet().to(DEV); opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEV == "cuda"))
    step, curve = 0, []
    if ck.exists():
        s = torch.load(ck, map_location=DEV)
        net.load_state_dict(s["net"]); opt.load_state_dict(s["opt"]); sch.load_state_dict(s["sch"])
        step, curve = s["step"], s["curve"]; print(f"  resuming from the checkpoint: step {step}")
    if step >= STEPS:
        print("  training already finished"); continue

    rng = np.random.default_rng(step)
    H, W = Xtr.shape[1:]
    def batch():
        idx = rng.integers(0, len(Xtr), BATCH); xs, ys, ms = [], [], []
        for i in idx:
            ok_ = np.argwhere(Mtr[i]); r, c = ok_[rng.integers(len(ok_))]
            r0 = int(np.clip(r - CROP // 2, 0, H - CROP)); c0 = int(np.clip(c - CROP // 2, 0, W - CROP))
            sl = (slice(r0, r0 + CROP), slice(c0, c0 + CROP))
            x, y, m_ = Xtr[i][sl].astype(np.float32), Ytr[i][sl].astype(np.float32), Mtr[i][sl]
            if rng.random() < 0.5: x, y, m_ = x[::-1], y[::-1], m_[::-1]
            if rng.random() < 0.5: x, y, m_ = x[:, ::-1], y[:, ::-1], m_[:, ::-1]
            xs.append(x.copy()); ys.append(y.copy()); ms.append(m_.copy())
        t = lambda a: torch.from_numpy(np.stack(a))[:, None].to(DEV)
        return t(xs) / SCALE, t(ys) / SCALE, t(ms).float()

    def validate():
        ps = [T.metrics(apply_net(net, x.astype(np.float32), SCALE), y.astype(np.float32), m_)["psnr"]
              for x, y, m_ in zip(Xva, Yva, Mva)]
        ps0 = [T.metrics(x.astype(np.float32), y.astype(np.float32), m_)["psnr"] for x, y, m_ in zip(Xva, Yva, Mva)]
        return float(np.mean(ps)), float(np.mean(ps0))

    t0 = time.time(); net.train(); run = 0.0
    while step < STEPS:
        x, y, m_ = batch()
        with torch.autocast(DEV, enabled=(DEV == "cuda")):
            loss = (((net(x) - y) ** 2) * m_).sum() / m_.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
        step += 1; run += float(loss)
        if step % VAL_EVERY == 0 or step == STEPS:
            vp, vp0 = validate(); net.train()
            curve.append(dict(dose=k, step=step, train_loss=run / VAL_EVERY, val_psnr=vp, val_psnr_input=vp0))
            run = 0.0
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sch=sch.state_dict(), step=step,
                            curve=curve, scale=SCALE), ck)
            pd.DataFrame(curve).to_csv(EG/f"curve_{k}.csv", index=False)
            print(f"  step {step:>6}/{STEPS}  loss {curve[-1]['train_loss']:.5f}  val PSNR {vp:.2f} dB "
                  f"(input {vp0:.2f})  {(time.time()-t0)/60:.1f} min", flush=True)
print("training done")
""")

md(r"""
## E · Test — per phantom, resumes where it stopped

For each test phantom and each trained model: OS-SART at 50 / 25 / 10 % from the MC projections →
U-Net; also 25 % synthetic (`A·sart_full` + the same dose model; the same seed as in 03). Measures:
PSNR / SSIM (`dataset_test.metrics`), correlation, correlation with `victre_fbp` where present (reported only),
mass and control SDNR. The measures of the input OS-SART on the same phantoms are recorded alongside.
""")
code(r"""
nets = {}
for k in TRAIN_DOSES:
    s = torch.load(EG/"ckpt"/f"unet_{k}.pt", map_location=DEV)
    n_ = UNet().to(DEV); n_.load_state_dict(s["net"]); nets[k] = (n_, s["scale"])
todo = [f for f in TEST if not (EG/"test"/f"{f.stem}.json").exists()]
print(f"test: {len(TEST)} phantoms, {len(TEST)-len(todo)} finished, {len(todo)} left", flush=True)
t_all = time.time()
for i, f in enumerate(todo):
    t = time.time()
    with np.load(f) as z: d = {kk: z[kk] for kk in z.files}
    dn, gt, m, valid, les = str(d["density"]), d["sart_full"], d["breast_mask"], d["proj_valid"], d["lesions"]
    geo = G.Geometry(dn, gt.shape[0])
    has_vf = bool(d["victre_fbp_present"])
    vf = (d["victre_fbp"].astype(np.float32) / 65535.0 * float(d["victre_fbp_scale"])) if has_vf else None
    pf = d["proj_full"].astype(np.float32); fp = pf > 0
    INP = {("mc", k): geo.os_sart(d[f"proj_{k}"].astype(np.float32), valid) for k in DOSES}
    want_ex = not (EG/"example.npz").exists() and bool(d["signal_present"]) and bool((les[:, 4] == 1).any())
    ps = np.where(fp & valid, np.clip(geo.project(gt), 0, None), 0.0).astype(np.float32)
    rng = np.random.default_rng(10_000 + int(f.stem.lstrip("-")) % 100_000)
    syn = {k: synth_dose(ps, fp, float(d["photons_per_pixel"]), float(d["electronic_sigma"]), float(k) / 100, rng) for k in DOSES}
    INP[("syn", "25")] = geo.os_sart(syn["25"], valid)
    rec = dict(patient=f.stem, density=dn, signal_present=int(bool(d["signal_present"])), victre_fbp_present=int(has_vf),
               metrics=[], sdnr=[])
    def score(name, train_dose, src, k, v):
        mm = T.metrics(v, gt, m)
        rec["metrics"].append(dict(model=name, train_dose=train_dose, source=src, dose=k, psnr=float(mm["psnr"]),
                                   ssim=float(mm["ssim"]), corr=corr_masked(v, gt, m),
                                   corr_vf=(corr_masked(v, vf, m) if has_vf else None)))
        for s0, r0, c0, num, kind in les:
            x = sdnr_at(v, s0, r0, c0)
            if x is not None:
                rec["sdnr"].append(dict(model=name, train_dose=train_dose, source=src, dose=k, kind=int(kind), sdnr=x))
    example = {}
    for (src, k), x in INP.items():
        score("OS-SART", "", src, k, x)
        for td, (net, sc) in nets.items():
            y = apply_net(net, x, sc); score("U-Net", td, src, k, y)
            if want_ex and src == "mc": example[f"unet{td}_{k}"] = y
        if want_ex and src == "mc": example[f"input_{k}"] = x
    if example:
        zm = int(round(float(les[les[:, 4] == 1][0, 0])))
        np.savez_compressed(EG/"example.npz", patient=f.stem, density=dn, z=zm, mask=m[zm], sart_full=gt[zm],
                            lesions=les, **{kk: v[zm] for kk, v in example.items()})
    rec["seconds"] = round(time.time() - t, 1)
    tmp = EG/"test"/f"{f.stem}.json.tmp"; tmp.write_text(json.dumps(rec, default=float)); tmp.rename(EG/"test"/f"{f.stem}.json")
    el = time.time() - t_all
    print(f"[{i+1}/{len(todo)}] {f.stem} {dn:<9} {rec['seconds']:.0f} s | ~{el/(i+1)*(len(todo)-i-1)/3600:.1f} h left", flush=True)
print("test done")
""")

md(r"""
### E2 · Example slice (`example.npz`, the U-Net panel of 02)

If E has finished and `example.npz` is missing (an earlier version looked only at the first phantom and wrote nothing
if that phantom had no lesion), produces only this file, for **the first test phantom with a mass**. ~1 min.
Does not change the metrics.
""")
code(r"""
if (EG/"example.npz").exists():
    print("example.npz already exists:", str(np.load(EG/"example.npz")["patient"]))
else:
    try: nets
    except NameError:
        nets = {}
        for k in TRAIN_DOSES:
            s = torch.load(EG/"ckpt"/f"unet_{k}.pt", map_location=DEV)
            n_ = UNet().to(DEV); n_.load_state_dict(s["net"]); nets[k] = (n_, s["scale"])
    for f in TEST:
        with np.load(f) as z:
            if not bool(z["signal_present"]) or not (z["lesions"][:, 4] == 1).any(): continue
            d = {kk: z[kk] for kk in z.files}
        break
    else:
        raise SystemExit("no test phantom with a mass")
    dn, gt, m, valid, les = str(d["density"]), d["sart_full"], d["breast_mask"], d["proj_valid"], d["lesions"]
    geo = G.Geometry(dn, gt.shape[0])
    zm = int(round(float(les[les[:, 4] == 1][0, 0])))
    example = {}
    for k in DOSES:
        x = geo.os_sart(d[f"proj_{k}"].astype(np.float32), valid)
        example[f"input_{k}"] = x[zm]
        for td, (net, sc) in nets.items():
            example[f"unet{td}_{k}"] = apply_net(net, x, sc)[zm]
    np.savez_compressed(EG/"example.npz", patient=f.stem, density=dn, z=zm, mask=m[zm], sart_full=gt[zm],
                        lesions=les, **example)
    print("example.npz written:", f.stem, dn, "z =", zm, "|", sorted(example))
""")

md("## F · Collect")
code(r"""
import pandas as pd
recs = [json.loads(p.read_text()) for p in sorted((EG/"test").glob("*.json"))]
assert recs, "no test phantoms -> run E first"
BASE = ["patient", "density", "signal_present", "victre_fbp_present"]
EM = pd.DataFrame([{**{k: r[k] for k in BASE}, **x} for r in recs for x in r["metrics"]])
ES = pd.DataFrame([{**{k: r[k] for k in BASE}, **x} for r in recs for x in r["sdnr"]])
EM.to_csv(RES/"training_metrics.csv", index=False); ES.to_csv(RES/"training_sdnr.csv", index=False)
CV = pd.concat([pd.read_csv(p) for p in sorted(EG.glob("curve_*.csv"))], ignore_index=True)
CV.to_csv(RES/"training_curve.csv", index=False)
print(f"{len(recs)} test phantoms | training_metrics {len(EM)} rows, training_sdnr {len(ES)}, curve {len(CV)}")
EM["train_dose"] = EM.train_dose.fillna("").astype(str); EM["dose"] = EM.dose.astype(str)
print("\nPSNR (dB) vs sart_full, MC source; row: model / training dose, column: test dose")
print(EM[EM.source == "mc"].pivot_table(index=["model", "train_dose"], columns="dose", values="psnr")[DOSES].round(2))
print("\n25 % synthetic:")
print(EM[EM.source == "syn"].groupby(["model", "train_dose"]).psnr.mean().round(2))
g = ES[(ES.source == "mc") & (ES.kind == -1)].groupby(["model", "train_dose", "dose"]).sdnr.mean()
print("\ncontrol SDNR (should be ~0):", g.round(3).to_dict())
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
