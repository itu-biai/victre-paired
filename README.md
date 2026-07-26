# VICTRE-Paired

Reproduction code for **VICTRE-Paired**, an open dataset for limited-angle
digital breast tomosynthesis (DBT) reconstruction. Each sample pairs the raw
Monte-Carlo projections of a virtual patient with the reconstructed volume, and
adds phantom-accurate lesion locations and multiple dose levels.

The dataset is derived from the public [VICTRE in-silico
trial](https://www.cancerimagingarchive.net/collection/victre/) (Badano et al.,
2018) and is intended as a benchmark for reconstruction research — distinct from
existing VICTRE-derived resources, which target detection and segmentation.

- **Data record:** [huggingface.co/datasets/yusuf-talha/victre-paired](https://huggingface.co/datasets/yusuf-talha/victre-paired) (DOI: _to be assigned_)
- **Paper:** _Data Descriptor, in preparation_

---

## What is in the dataset

| Property | Value |
|---|---|
| Patients | 2761 (train 2208 / val 276 / test 277) |
| Chunks | 346 `.npz` files, 8 patients each |
| Projections | 25 views, ±25°, 752 × 384 (Monte-Carlo, flat-field corrected) |
| Reconstruction | FBP volume, 56 × 408 × 336, 0.34 × 0.34 × 1.0 mm |
| Dose levels | full / half / quarter (Poisson-Gaussian) |
| Lesions | phantom-accurate coordinates, mass and calcification types |
| Densities | fatty, scattered, heterogeneous, dense |
| License | data: CC BY 4.0 · code: MIT |

Two reconstruction regimes are supported (see [Two regimes](#two-regimes-inverse-crime)):
an **inverse-crime** regime driven by synthetic forward projections `A(clean)`, and an
**inverse-crime-free** regime driven by the real Monte-Carlo projections `clean_proj`.

### Chunk contents (`.npz` keys)

Each chunk holds 8 patients (first axis). Reconstruction arrays are
`(8, 56, 408, 336)`, projection arrays are `(8, 25, 752, 384)`.

| Key | Shape | Description |
|---|---|---|
| `clean` | (8, 56, 408, 336) | FBP reconstruction (reference), 99.5-percentile normalized |
| `clean_proj` | (8, 25, 752, 384) | real MC projections (attenuation), aligned |
| `noisy_proj` | (8, 25, 752, 384) | noisy projections, **half dose** (default) |
| `noisy_proj_full` / `noisy_proj_quarter` | (8, 25, 752, 384) | full / quarter dose |
| `noisy` | (8, 56, 408, 336) | noisy reconstruction (half dose) |
| `mask` | (8, 56, 408, 336) | breast mask |
| `lesion_coords` | (8, 8, 4) | up to 8 lesions per patient: `[z, h, w, type]` |
| `lesion_count` | (8,) | number of valid lesions |
| `is_pos` | (8,) | signal-present flag |
| `seed`, `density`, `native_z` | (8,) | patient id, breast density, native slice count |
| `sigma`, `sigma_full`, `sigma_quarter`, `sigma_recon` | (8,) | measured noise std |
| `recon_scale`, `proj_scale` | (8,) | per-patient normalization factors |
| `dose_levels`, `dose_gains`, `elec_noise`, `strip_valley` | — | global parameters |

Lesion `type` follows VICTRE: 0–3 are microcalcification clusters, 4–7 are masses.
**Masses are resolvable; microcalcifications are not** at this resolution (see
`docs`/limitations) — use masses for lesion-based tasks.

---

## Repository layout

```
victre-paired-v6/
├── README.md
├── requirements.txt
├── LICENSE
└── src/
    ├── constants.py            shared geometry / dose / split parameters
    ├── generate_dataset.py     build the dataset from VICTRE source data
    ├── validate_dataset.py     technical validation (integrity + physics)
    ├── geometry.py             LEAP forward / adjoint operators
    ├── run_baselines.py        two-regime baselines (FBP, ATp, SIRT, SART, ASD-POCS)
    │                           + masked correlation, scale-matched PSNR, background energy
    └── figures/
        ├── make_dataset_figures.py    dataset characterization figures
        └── make_baseline_figures.py   baseline tables and figures
```

---

## Installation

Python ≥ 3.10. A CUDA GPU is required for the reconstruction operators
(`geometry.py`, `run_baselines.py`); generation and validation of most metrics
run on CPU.

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
clean      = d["clean"]        # (8, 56, 408, 336) reference reconstruction
clean_proj = d["clean_proj"]   # (8, 25, 752, 384) real MC projections
noisy_proj = d["noisy_proj"]   # (8, 25, 752, 384) half-dose noisy projections

# lesions of the first patient
n = int(d["lesion_count"][0])
lesions = d["lesion_coords"][0][:n]   # rows [z, h, w, type]
masses  = lesions[lesions[:, 3] >= 4]
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
adapted to your environment. The split is deterministic (`SEED_SPLIT = 42`).

> Note: the released dataset seeds the noise generator once per session. For
> strictly bit-reproducible noise, seed per patient by SEED (see the noise model
> in `generate_dataset.py`); the paired projections and reconstructions are fully
> deterministic regardless.

## Technical validation

```bash
python src/validate_dataset.py
```

checks file integrity, split disjointness, metadata consistency, the dose model
(monotonic noise across dose levels), projection uniformity, and lesion
conspicuity, and writes a JSON/Markdown report. This reproduces the figures and
numbers in the Technical Validation section of the paper.

## Baselines

```bash
python src/run_baselines.py            # writes per-patient metrics for the test split
python src/figures/make_baseline_figures.py
```

`run_baselines.py` reconstructs every test patient with nine classical methods
(FBP; Aᵀp; SIRT-20/50/100; SART-2/4/8; ASD-POCS-20) under **both** regimes.

FBP is implemented from scratch (Hann-windowed ramp filter, edge-replicate
padding, approximate cosine weighting) rather than adapted from VICTRE's own
GPL-licensed reconstruction code, which is neither read nor copied here. Two
built-in LEAP FBP routes were tried and rejected during development: `L.FBP()`
returns a degenerate result on this modular-beam geometry, and
`L.filterProjections()` is a silent no-op on this geometry (its output is
numerically identical to unfiltered back-projection) — a useful reminder to
sanity-check library routines against a known baseline (e.g. Aᵀp) before
trusting their output.

**Metrics and metric hierarchy.** Six metrics are recorded per reconstruction:
breast-masked correlation (`corr_mask`, **primary** cross-regime metric),
whole-volume correlation (`corr`, diagnostic), raw and scale-matched PSNR,
global SSIM, background energy (`bg`, diagnostic), and mass SDNR.

Whole-volume correlation is confounded by background behaviour: FBP's ramp
filter leaks energy into the zero-padded regions of the projections (dead
detector frame, penumbra strip), inflating background intensity, while Aᵀp
drives the background to near zero "for free". Since background voxels
dominate the volume by count, whole-volume correlation rewards a clean
background over faithful breast reconstruction — in this dataset it can even
invert the ranking of methods relative to what is visible inside the breast.
`corr_mask`, computed only inside the breast mask, is the metric reported as
primary in the paper; `corr` and `bg` are kept as diagnostics.

---

## Two regimes (inverse crime)

The dataset can drive reconstruction from two projection sources:

- **`A(clean)` — inverse-crime regime.** Synthetic forward projections computed
  by the *same* line-integral operator used inside the reconstructors. By
  construction this is an *inverse crime* (Kaipio & Somersalo, 2005): a
  best-case, geometry-matched setting for controlled method development.
- **`clean_proj` — inverse-crime-free regime.** The real Monte-Carlo projections,
  which contain scatter, beam hardening and detector response that the operator
  does not model — a realistic challenge for mismatch-robust, scatter-corrected
  and self-supervised reconstruction.

The mismatch is quantified in the paper: iterative reconstruction *improves* under
`A(clean)` but *degrades* under `clean_proj` (the signature of forward-model
mismatch), and the ideal–real gap grows monotonically with a method's reliance on
the forward operator — from FBP (a single filtered back-projection, no
data-consistency loop, gap ≈ 0.09 in breast-masked correlation) to SIRT-100 (100
enforcements, gap ≈ 0.73). Report both regimes when benchmarking; the gap
measures a method's robustness to model mismatch.

Notably, FBP's *whole-volume* correlation is essentially unchanged between
regimes (real ≳ ideal): because `clean` is itself produced by filtering and
back-projecting the real Monte-Carlo projections, applying FBP to those same
projections approximately retraces that production path — an incidental,
partial check that the projection pipeline (flat-field estimate, alignment
offsets, geometry) is self-consistent with how the reference volume was made.

---

## Known limitations (summary)

- The reference (`clean`) is an FBP reconstruction used as a "so-called ground
  truth", following LoDoPaB-CT (Leuschner et al., Sci. Data 2021).
- The flat-field is estimated (the true VICTRE flat-field is not published).
- Alignment offsets are empirical; a small residual mismatch remains.
- Microcalcifications are below the resolving limit after 4× downsampling; use
  masses for lesion tasks.
- One patient (SEED 208084664) has a degenerate VICTRE reconstruction; it is kept
  for completeness and excluded in evaluation via `BROKEN_SEEDS`.

See the paper's Usage Notes / limitations for the full discussion.

---

## Citation

If you use this dataset or code, please cite the Data Descriptor (details to
follow) and the VICTRE trial:

> Badano A, et al. Evaluation of Digital Breast Tomosynthesis as Replacement of
> Full-Field Digital Mammography Using an In Silico Imaging Trial. JAMA Network
> Open, 2018.

## License

Code is released under the MIT License (`LICENSE`). The dataset is released under
CC BY 4.0. VICTRE source data is distributed by the NCI Cancer Imaging Archive
under CC BY 3.0.
