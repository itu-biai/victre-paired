# VICTRE-Paired

Reproduction code for **VICTRE-Paired**, an open dataset for limited-angle
digital breast tomosynthesis (DBT) reconstruction. Each sample pairs the 25 raw
Monte-Carlo projections of a virtual patient with the reconstructed volume, and
adds phantom-accurate lesion locations, control regions, three dose levels, and
the full projection geometry needed to run a forward/adjoint operator.

The dataset is derived from the public [VICTRE in-silico
trial](https://www.cancerimagingarchive.net/collection/victre/) (Badano et al.,
2018) and is intended as a benchmark for **reconstruction** research — distinct
from existing VICTRE-derived resources, which target detection and segmentation.

- **Data record:** [huggingface.co/datasets/yusuf-talha/victre-paired](https://huggingface.co/datasets/yusuf-talha/victre-paired) (DOI: _to be assigned_)
- **Paper:** _Data Descriptor, in preparation_

---

## What is in the dataset

| Property | Value |
|---|---|
| Patients | 2761 (train 2208 / val 276 / test 277) |
| Chunks | 348 `.npz` files, 8 patients each |
| Projections | 25 views, ±25°, 752 × 384 (Monte-Carlo, flat-field corrected) |
| Reconstruction | FBP volume, 56 × 408 × 336, 0.34 × 0.34 × 1.0 mm |
| Dose levels | full / half / quarter (intensity-domain Poisson + electronic) |
| Lesions | phantom-accurate coordinates; mass and calcification types |
| Control ROIs | matched signal-absent regions in negative patients |
| Densities | fatty, scattered, heterogeneous, dense |
| Size | ~115 GB |
| License | data: CC BY 4.0 · code: MIT · source: CC BY 3.0 |

Two reconstruction regimes are supported (see [Two regimes](#two-regimes-inverse-crime)):
an **inverse-crime** regime driven by synthetic forward projections `A(clean)`,
and an **inverse-crime-free** regime driven by the real Monte-Carlo projections
`clean_proj`.

### Chunk contents (`.npz` keys)

Each chunk holds 8 patients (first axis). Reconstruction arrays are
`(8, 56, 408, 336)`, projection arrays are `(8, 25, 752, 384)`. Volumes,
projections and mask are stored as `uint16`/`uint8` to save space; divide by
`65535` (or `255` for the mask) to recover the `[0, 1]` range.

| Key | Shape | dtype | Description |
|---|---|---|---|
| `clean` | (8, 56, 408, 336) | uint16 | FBP reconstruction (reference), 99.5-percentile normalized |
| `clean_proj` | (8, 25, 752, 384) | uint16 | real MC projections, `log(ff) − log(I)` domain, **not shifted** |
| `mask` | (8, 56, 408, 336) | uint8 | breast mask = `clean > 0.08` |
| `noisy_proj` | (8, 25, 752, 384) | uint16 | noisy projections, **half dose** (default) |
| `noisy_proj_full` / `noisy_proj_quarter` | (8, 25, 752, 384) | uint16 | full / quarter dose |
| `sigma` / `sigma_full` / `sigma_quarter` | (8,) | float32 | measured noise std per dose |
| `is_pos` | (8,) | bool | signal-present flag (from `.loc` file type) |
| `seed` | (8,) | int64 | VICTRE phantom SEED (patient id) |
| `density` | (8,) | str | fatty / scattered / heterogeneous / dense |
| `native_z` / `native_x` / `native_y` | (8,) | int16 | native reconstruction dimensions (class-constant) |
| `lesion_coords` | (8, 8, 4) | float32 | up to 8 lesions: `[z, h, w, type]` |
| `lesion_count` | (8,) | int16 | number of valid lesions |
| `control_rois` | (8, 12, 5) | float32 | control regions: `[z, h, w, roi_id, in_bounds]` |
| `control_count` | (8,) | int16 | number of valid control ROIs |
| `recon_scale` / `proj_scale` | (8,) | float32 | per-patient normalization factors |
| `noise_seed` | (8,) | int64 | per-patient noise seed (= `seed`) |
| `geom_vox_z` | (8,) | float32 | voxel z-size = `native_z / 56` |
| `geom_offx` / `geom_offy` / `geom_offz` | (8,) | float32 | volume offsets for the projection geometry |
| `strip_valley` | (8,) | float32 | per-patient penumbra-strip threshold |
| `sid` / `sdd` / `det_pix` | () | float32 | 600.0 / 650.0 / 0.34 (constant across the dataset) |
| `elec_noise` | () | float32 | electronic noise std (0.010) |
| `dose_levels` / `dose_gains` | (3,) | float32 | `[1, 0.5, 0.25]` / `[0.001, 0.002, 0.004]` |

Lesion `type` follows VICTRE: 0–3 are microcalcification clusters, 4–7 are masses.
**Masses are resolvable; microcalcifications are not** at this resolution (see
[Known limitations](#known-limitations-summary)) — use masses for lesion-based tasks.

> **`clean_proj` is not pre-aligned.** Unlike a rectified image stack, the
> projections are stored in the raw detector frame. To relate them to `clean`
> you must build the projection geometry from the per-patient `geom_*` fields
> (see [Using the dataset](#using-the-dataset)). Feeding `clean_proj` to a
> reconstructor without this geometry will not align with `clean`.

---

## Repository layout

```
victre-paired/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
└── src/
    ├── constants.py            shared geometry / dose / split parameters
    ├── generate_dataset.py     build the dataset from VICTRE source data
    ├── validate_dataset.py     technical validation (integrity + physics + baselines)
    ├── geometry.py             LEAP forward / adjoint operators from geom_* fields
    ├── run_baselines.py        two-regime reconstruction (9 methods) -> baseline_raw.csv
    └── figures/
        ├── make_baseline_figures.py   baseline_raw.csv -> Table 3 + F5/F5b/F5c/F6
        └── make_dataset_figures.py    validate_dataset.py's tables -> F1/F2/F3/F4/F6/F7
```

`run_baselines.py` only reconstructs and scores (needs a GPU); the two
`figures/` scripts turn its CSV output, and `validate_dataset.py`'s, into the
paper's tables and figures and do not need a GPU themselves (except for
`make_baseline_figures.py`'s optional F6 gallery, which re-runs a few
reconstructions for illustration).

---

## Installation

Python ≥ 3.10. A CUDA GPU is required for the reconstruction operators
(`geometry.py`, `run_baselines.py`); dataset generation and most validation
metrics run on CPU.

```bash
pip install -r requirements.txt
```

The reconstruction operators use [LEAP](https://github.com/LLNL/LEAP)
(LivermorE AI Projector), installed from source:

```bash
git clone --depth 1 https://github.com/LLNL/LEAP.git
pip install ./LEAP
```

---

## Using the dataset

```python
import numpy as np

d = np.load("test/test_chunk_00000.npz")
clean      = d["clean"].astype(np.float32) / 65535      # (8, 56, 408, 336) reference
clean_proj = d["clean_proj"].astype(np.float32) / 65535 # (8, 25, 752, 384) real MC projections
noisy_proj = d["noisy_proj"].astype(np.float32) / 65535 # (8, 25, 752, 384) half-dose

# lesions of the first patient
n       = int(d["lesion_count"][0])
lesions = d["lesion_coords"][0][:n]        # rows [z, h, w, type]
masses  = lesions[lesions[:, 3] >= 4]      # type >= 4 → masses
```

### Building the projection geometry

`clean_proj` lives in the raw detector frame; to reconstruct or forward-project
you build a LEAP modular-beam geometry from the stored per-patient fields. The
constants `SID = 600`, `SDD = 650`, `DET_PIX = 0.34` and the ±25° / 25-view
trajectory are the same for every patient; only the volume placement
(`geom_vox_z`, `geom_offx/y/z`) varies. `src/geometry.py` wraps this:

```python
from src.geometry import Geometry

i = 0                                        # patient within the chunk
G = Geometry(vox_z=float(d["geom_vox_z"][i]),
             offx=float(d["geom_offx"][i]),
             offy=float(d["geom_offy"][i]),
             offz=float(d["geom_offz"][i]))
recon = G.fbp(clean_proj[i])                 # or G.sirt(clean_proj[i], n=50), G.atp(...)
```

### Reproducing the stored noise

Noise is generated in the intensity domain and is bit-exactly reproducible from
`noise_seed`:

```python
def add_noise(p, proj_scale, gain, noise_seed, dose_idx, s_elec=0.010):
    r  = np.random.default_rng(np.uint64(noise_seed) * 10 + np.uint64(dose_idx))
    I0 = 1.0 / gain
    I  = I0 * np.exp(-np.clip(p, 0, 1) * proj_scale)     # transmitted photons
    N  = r.poisson(np.maximum(I, 1e-9)).astype(np.float64)
    N += r.standard_normal(p.shape) * (s_elec * I0 * 0.02)
    return np.clip(-np.log(np.maximum(N, 1e-9) / I0) / proj_scale, 0, 1)

# dose_idx: 0 = full, 1 = half, 2 = quarter
```

---

## Reproducing the dataset

Generation is designed to run across five Google Colab sessions in parallel
(the train split is produced in three shards). In `src/generate_dataset.py` set

```python
RUN_MODE = "train_1"   # then "train_2", "train_3", "val", "test" in other sessions
```

and run. Completed chunks are skipped, so interrupted runs can be restarted.
Paths at the top of the script point to a Google Drive mount and should be
adapted to your environment. The split is deterministic (`SEED_SPLIT = 42`) and
read from the VICTRE source assignment — it is not re-shuffled. All noise is
seeded per patient and per dose, so the entire dataset is reproducible
regardless of run order or interruptions.

---

## Technical validation

```bash
python src/validate_dataset.py --data /path/to/victre-paired --out ./validation_report --profile quick
python src/figures/make_dataset_figures.py --validation ./validation_report
```

`validate_dataset.py` checks, over the full population where cheap and on a
stratified sample for the heavy per-array measurements:

- **Integrity** — schema, dtypes, shapes, constants, no NaN/Inf, no duplicate
  seeds, split disjointness, and the analytic geometry formula (all 2761 patients).
- **Normalization** — `clean` 99.5-percentile = 1.0, `clean_proj` 99.8-percentile = 1.0.
- **Flat-field** — air attenuation ≈ 0 (physical requirement), angular uniformity
  ≈ 1/cos 25° = 1.10.
- **Noise** — measured vs. stored `sigma`, dose monotonicity, whiteness, and
  **bit-exact reproduction** of every `noisy_proj` array from `noise_seed`.
- **Geometry** — residual parallax between `A(clean)` and `clean_proj` (needs a GPU).
- **Task-based** — mass/control SDNR, d′, AUC; unbiased-estimator check on control ROIs.

It writes `report.md` / `results.json` reproducing the numbers in the paper's
Technical Validation section, plus per-patient tables under `tables/`.
`figures/make_dataset_figures.py` turns those tables into the paper's F1–F4
figures (and F6/F7 if stages 7/9 were run) and does not itself need a GPU.

---

## Baselines

```bash
python src/run_baselines.py --data /path/to/victre-paired --out ./paper --split test
python src/figures/make_baseline_figures.py --out ./paper
```

`run_baselines.py` reconstructs every test patient with **nine** classical
methods (FBP; Aᵀp; SIRT-20/50/100; SART-2/4/8; ASD-POCS-20) under **both**
regimes and writes per-patient rows to `paper/tables/baseline_raw.csv`; it
needs a GPU and is resumable (safe to interrupt and re-run). It does not
itself produce tables or figures.

`figures/make_baseline_figures.py` reads that CSV and writes `paper/tables/`
(`baseline_summary.csv`, `inversion_stats.json`, `T3_baseline.tex`/`.md`) and
`paper/figures/` (`F5_two_regime`, `F5b_density`, `F5c_task_sdnr`, and, if you
pass `--data`, the `F6_gallery_*` reconstruction gallery). This step needs no
GPU except for the optional F6 gallery.

FBP is implemented from scratch (Hann-windowed ramp filter, edge-replicate
padding, approximate cosine weighting) rather than adapted from VICTRE's own
GPL-licensed reconstruction code, which is neither read nor copied here. Two
built-in LEAP FBP routes were tried and rejected during development: `L.FBP()`
returns a degenerate result on this modular-beam geometry, and
`L.filterProjections()` is a silent no-op here (its output is numerically
identical to unfiltered back-projection) — a reminder to sanity-check library
routines against a known baseline (e.g. Aᵀp) before trusting them.

**Metrics.** Per reconstruction: breast-masked correlation (**primary**),
scale-matched PSNR, SSIM, RMSE, and mass SDNR. Whole-volume PSNR/correlation are
confounded by background behaviour — FBP's ramp filter leaves a constant offset
across the zero-padded detector frame, which depresses whole-volume PSNR and
SSIM even though the breast texture is the most faithful of all methods. Report
masked correlation and scale-matched PSNR; the raw variants are misleading here.

---

## Two regimes (inverse crime)

The dataset can drive reconstruction from two projection sources:

- **`A(clean)` — inverse-crime regime.** Synthetic forward projections computed
  by the *same* line-integral operator used inside the reconstructors. By
  construction this is an *inverse crime* (Kaipio & Somersalo, 2005): a best-case,
  geometry-matched setting for controlled method development.
- **`clean_proj` — inverse-crime-free regime.** The real Monte-Carlo projections,
  which contain scatter, beam hardening and detector response that the operator
  does not model — a realistic challenge for mismatch-robust, scatter-corrected
  and self-supervised reconstruction.

The mismatch reverses the method ranking. Under `A(clean)`, iterative methods win
and FBP is last (SIRT-100 best, FBP worst by masked correlation). Under the real
`clean_proj`, **the ranking inverts at both ends**: FBP becomes the single best
method — best in **276/276** test patients — while the most heavily iterated
method (SIRT-100) becomes worst. FBP's inverse-crime gap is *negative* in all
276 patients; every iterative method's gap is large and positive. The eight
non-FBP methods collapse into a narrow band (masked correlation ≈ 0.43–0.45) in
the real regime, so their relative order there is noise; the finding is the
top/bottom swap, not a smooth monotone trend. **Report both regimes when
benchmarking** — the gap measures a method's robustness to forward-model mismatch.

Two mechanisms explain FBP's real-regime advantage, and they point the same way.
First, `clean` is itself produced by filtering and back-projecting the real
Monte-Carlo projections, so applying FBP to those same projections partially
retraces the reference's own production path. Second, the ramp filter has zero
DC gain, so FBP is insensitive to the flat-field pedestal that corrupts every
iterative solver (a controlled DC-injection sweep confirms this: FBP is
essentially immune, iterative solvers degrade with pedestal size and iteration
count). No classical method reaches ground-truth lesion contrast (mass SDNR
0.26–0.51 vs. 0.70 for the reference), which is precisely the headroom a learned
reconstructor can target.

---

## Known limitations (summary)

- The reference (`clean`) is an FBP reconstruction used as a "so-called ground
  truth", following LoDoPaB-CT (Leuschner et al., Sci. Data 2021).
- The flat-field is estimated (the true VICTRE flat-field is not published);
  the residual air pedestal is measured and reported in the paper.
- Microcalcifications are below the resolving limit at this resolution; use
  masses for lesion tasks.
- Class balance is inherited from VICTRE (fatty is the smallest class).
- Dose labels are defined by an absolute photon budget (`I0 = 1/gain`), not
  calibrated to clinical mGy.
- One patient (SEED 208084664) has a degenerate VICTRE reconstruction; it is
  kept for completeness and excluded in evaluation via `BROKEN_SEEDS`.

See the paper's Usage Notes / limitations for the full discussion.

---

## Citation

If you use this dataset or code, please cite the Data Descriptor (details to
follow) and the VICTRE trial:

> Badano A, et al. Evaluation of Digital Breast Tomosynthesis as Replacement of
> Full-Field Digital Mammography Using an In Silico Imaging Trial. JAMA Network
> Open, 2018.

## License

Code is released under the MIT License (`LICENSE`). The dataset is released
under CC BY 4.0. VICTRE source data is distributed by the NCI Cancer Imaging
Archive under CC BY 3.0.
