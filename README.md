# VICTRE-Paired

Reproduction code for **VICTRE-Paired**, an open dataset for limited-angle
digital breast tomosynthesis (DBT) reconstruction. Each sample pairs the 25 raw
Monte-Carlo projections of a virtual patient with the reconstructed volume, and
adds phantom-accurate lesion locations, control regions, three photon budgets, and
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
| Chunks | `.npz` files named `{split}_s{shard}_chunk_{NNNNN}.npz`, up to 8 patients each (a few shard-final chunks hold fewer) |
| Projections | 25 views, ±25°, 752 × 384 (Monte-Carlo, corrected with an estimated flat field — see Flat-field below) |
| Reconstruction | FBP volume, 56 × 408 × 336; 0.34 × 0.34 mm in-plane, slice thickness `native_z / 56` mm (0.68–1.11 mm by density) |
| Photon budgets | three — 1,000 / 500 / 250 photons per 0.34 mm pixel per view (`full` / `half` / `quarter`); Poisson + 2-photon Gaussian, generated from the released `clean_proj`. These are relative noise levels, not calibrated clinical doses |
| Lesions | phantom-accurate coordinates; mass and calcification types |
| Control ROIs | matched signal-absent regions in negative patients |
| Densities | fatty, scattered, heterogeneous, dense |
| License | data: CC BY 4.0 · code: MIT · source: CC BY 3.0 |

`format_version` in every chunk identifies the data format and `changes_from_v8`
carries a one-line summary of what that format changed, so a loaded record states
its own provenance without reference to anything outside it.

Two reconstruction regimes are supported (see [Two regimes](#two-regimes-inverse-crime)):
an **inverse-crime** regime driven by synthetic forward projections `A(clean)`,
and an **inverse-crime-free** regime driven by the real Monte-Carlo projections
`clean_proj`.

### Chunk contents (`.npz` keys)

Each chunk holds up to 8 patients (first axis). Reconstruction arrays are
`(8, 56, 408, 336)`, projection arrays are `(8, 25, 752, 384)`. `clean` and
`clean_proj` are stored as `uint16`; divide by `65535` to recover the `[0, 1]`
range. `mask` is `uint8` with values 0 or 1 (no scaling). The **noisy projections use
a different encoding**: `p = (u16 − 5000) × 5e-5`, i.e. the range
`noisy_range = [−0.25, 3.02675]`, so that air can be negative (no clipping
pedestal) and the thickest tissue cannot saturate. Values outside the
illuminated detector area are stored exactly at the zero level 5000.

| Key | Shape | dtype | Description |
|---|---|---|---|
| `clean` | (8, 56, 408, 336) | uint16 | FBP reconstruction (reference), normalised by its **maximum** — nothing is clipped |
| `clean_proj` | (8, 25, 752, 384) | uint16 | real MC projections, `log(ff) − log(I)` domain, normalised by its **maximum**, **not shifted** |
| `mask` | (8, 56, 408, 336) | uint8 | breast mask = `clean > 0.08 * p99.5(clean)` (see below) |
| `noisy_proj_full` / `noisy_proj_half` / `noisy_proj_quarter` | (8, 25, 752, 384) | uint16 | noisy projections at 1,000 / 500 / 250 photons. **Decode with `(u16 − 5000) × 5e-5`**, not `/65535` |
| `sigma_full` / `sigma_half` / `sigma_quarter` | (8,) | float32 | measured noise std per budget, `std(noisy − clean)` on `clean_proj > 0.02` |
| `is_pos` | (8,) | bool | signal-present flag (from `.loc` file type) |
| `seed` | (8,) | int64 | VICTRE phantom SEED (patient id) |
| `density` | (8,) | str | fatty / scattered / heterogeneous / dense |
| `native_z` / `native_x` / `native_y` | (8,) | int16 | native reconstruction dimensions (class-constant) |
| `lesion_coords` | (8, 8, 4) | float32 | up to 8 lesions: `[z, h, w, type]` |
| `lesion_count` | (8,) | int16 | number of valid lesions |
| `control_rois` | (8, 12, 5) | float32 | control regions: `[z, h, w, roi_id, in_bounds]` |
| `control_count` | (8,) | int16 | number of valid control ROIs |
| `recon_scale` / `proj_scale` | (8,) | float32 | per-patient normalisation factors; each is the **maximum** of the corresponding array |
| `noise_seed` | (8,) | int64 | per-patient noise seed (= `seed`); rng = `default_rng(noise_seed·10 + dose_idx)` |
| `geom_vox_z` | (8,) | float32 | voxel z-size = `native_z / 56` |
| `geom_offx` / `geom_offy` / `geom_offz` | (8,) | float32 | volume offsets for the projection geometry |
| `recon_source_saturated` | (8,) | float32 | fraction of voxels at the uint16 ceiling in the **source** reconstruction DICOMs — the archive's own clipping, not this pipeline's |
| `sid` / `sdd` / `det_pix` | () | float32 | 600.0 / 650.0 / 0.34 (constant across the dataset) |
| `elec_noise_photons` | () | float32 | Gaussian electronic noise std, in photons (2.0) |
| `noisy_range` | (2,) | float32 | `[−0.25, 3.02675]`: decoded range of the `noisy_proj*` arrays |
| `dose_levels` / `dose_gains` | (3,) | str / float32 | `["full", "half", "quarter"]` / `[0.001, 0.002, 0.004]` |
| `format_version` | () | str | data format identifier |
| `changes_from_v8` | () | str | one-line summary of what this format changed, so a loaded chunk states its own provenance |

**The mask threshold is a formula, not a constant.** `MASK_THR = 0.08` is applied
to the *normalised* volume, so its physical level depends on the normalisation
divisor. Because this format divides by the maximum rather than a percentile, a
literal 0.08 would sit about 1.8x higher than in the previous format; measured, it
removed 14.3 % of the mask volume on average (20.4 % worst, concentrated in dense
breasts). The threshold is therefore anchored to the same percentile:

```python
mask = clean > 0.08 * np.percentile(clean, 99.5)   # percentile over the WHOLE volume
```

recomputable from the released array alone, so no extra field is stored.

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
    ├── constants.py            shared geometry / noise / split parameters
    ├── noise.py                noise model: regenerate / encode / decode noisy_proj*
    ├── estimate_flatfield.py   estimate the flat field from air regions -> flatfield/
    ├── fit_delta.py            re-fit the per-density z offset and report the residual
    ├── evaluate.py             THE reference evaluation for anyone reusing the data
    ├── generate_dataset.py     build the dataset from VICTRE source data
    ├── export_splits.py        write splits.csv (seed,split) from a released copy
    ├── splits.csv              released train/val/test membership (read by generate_dataset.py)
    ├── validate_dataset.py     technical validation (integrity + physics + baselines)
    ├── geometry.py             LEAP forward / adjoint operators from geom_* fields
    │                            `geometry_from_chunk(d, i)` builds one directly
    │                            from a loaded chunk's geom_* arrays
    ├── run_baselines.py        two-regime reconstruction (9 methods) -> baseline_raw.csv
    └── figures/
        ├── make_baseline_figures.py   baseline_raw.csv -> T3_baseline + F5/F5b/F5c/F6
        └── make_dataset_figures.py    validate_dataset.py's tables -> F1/F2/F3/F4/F6/F7
```

**Reusing the dataset?** `src/evaluate.py` is the entry point: give it a directory
of `<seed>.npy` reconstructions and it reports the same metrics the paper reports,
computed the same way — agreement after one documented affine fit inside the breast
mask, and detection (AUC, d', control SDNR) on the matched positions, with intervals
bootstrapped over positions. Numbers produced any other way are not comparable to
the paper's baseline table.

```bash
python src/evaluate.py --data /path/to/victre-paired --pred ./my_recons --name my-method
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

d = np.load("test/test_s0_chunk_00000.npz")
clean      = d["clean"].astype(np.float32) / 65535      # (8, 56, 408, 336) reference
clean_proj = d["clean_proj"].astype(np.float32) / 65535 # (8, 25, 752, 384) real MC projections

# the noisy arrays use their OWN quantisation -- /65535 here would be wrong
noisy_proj = (d["noisy_proj_half"].astype(np.float32) - 5000) * 5e-5   # 500 photons

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

Noise is generated in the intensity domain **from the released `clean_proj`**, so
every `noisy_proj*` array regenerates bit-for-bit from the record (`src/noise.py`):

```python
from src.noise import regenerate, decode
u16 = regenerate(d["clean_proj"][i], float(d["proj_scale"][i]), int(d["seed"][i]), "half")
assert np.array_equal(u16, d["noisy_proj_half"][i])
noisy = decode(d["noisy_proj_half"][i])       # float32 log-attenuation, (25, 752, 384)
```

Model: `I0 = 1/gain` photons per pixel per view (1 000 / 500 / 250),
`I = I0·exp(−p·proj_scale)`, `N ~ Poisson(I) + N(0, 2 photons)`,
`p_noisy = −log(max(N, 0.5)/I0)/proj_scale` inside the illuminated bounding
box of `clean_proj > 0`; rng `default_rng(seed·10 + dose_idx)`, dose_idx
full:0 half:1 quarter:2. The budgets are relative noise levels, roughly two
orders of magnitude below clinical air-side counts, chosen to give visible noise
at this pixel size.

---

## Reproducing the dataset

Generation runs per split and can be sharded across machines:

```bash
python src/generate_dataset.py --split train --shard 0 --n-shards 3   # and shards 1, 2
python src/generate_dataset.py --split val
python src/generate_dataset.py --split test
```

Completed chunks are skipped, so interrupted runs can be restarted. Paths at the
top of the script (`SOURCE_ROOT`, `LOC_ROOT`, `FLATFIELD_DIR`, `OUTPUT_ROOT`)
should be adapted to your environment.

**Split assignment.** The released train/val/test membership is a fixed
patient-level list, `splits.csv` (`seed,split`), which `generate_dataset.py`
reads when present next to `constants.py`. It can be regenerated from a copy of
the released dataset with `python src/export_splits.py --data /path/to/victre-paired`.
Without `splits.csv` the script falls back to a deterministic 80/10/10 hash of
the seed, which does **not** reproduce the released membership.

**Flat-field.** The true VICTRE flat-field is not published. Generation needs
two estimated inputs in `FLATFIELD_DIR`: `ff_base.npy` (per-view base
flat-field, 25 × 3000 × 1500) and `coefficients.json` (per-density, per-view
polynomial correction). Their estimation is described in the paper (Methods,
Projections); the files used for the released record accompany the data
repository.

A third file, `valley.json`, was read by the previous format: a per-density
attenuation threshold above which pixels were zeroed, on the assumption that they
were a collimator penumbra. They were not — they were the thickest tissue against
the chest wall. The threshold is gone and the file is no longer read.

All noise is seeded per patient and per dose, so the noisy arrays are
reproducible regardless of run order or interruptions.

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
- **Normalisation** — `clean` and `clean_proj` are each divided by their own
  maximum and nothing is clipped, so exactly one sample per patient sits at the
  uint16 ceiling by construction. Recover physical units with `recon_scale` /
  `proj_scale`.
- **Flat-field** — air attenuation ≈ 0 (physical requirement), angular uniformity
  ≈ 1/cos 25° = 1.10.
- **Noise** — measured vs. stored `sigma`, dose monotonicity, whiteness, and
  **bit-exact reproduction** of every `noisy_proj*` array from `noise_seed`.
- **Geometry** — residual parallax between `A(clean)` and `clean_proj` (needs a GPU).
- **Task-based** — mass/control SDNR, d′, AUC; unbiased-estimator check on control ROIs.

It writes `report.md` / `results.json` reproducing the numbers in the paper's
Technical Validation section, plus per-patient tables under `tables/`.
`figures/make_dataset_figures.py` turns those tables into the validation figures
F1–F4 (and F6/F7 if stages 7/9 were run) and does not itself need a GPU.

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

FBP is implemented in `geometry.py` (Hann-windowed ramp filter, edge-replicate
padding, approximate cosine weighting, back-projection through LEAP's adjoint)
rather than adapted from VICTRE's own GPL-licensed reconstruction code, which is
neither read nor copied here. LEAP's built-in `FBP()` / `filterProjections()`
are not used: on this modular-beam geometry they did not produce a usable
filtered reconstruction in our hands.

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
- The three noise levels are photon budgets (1 000 / 500 / 250 per 0.34 mm
  pixel per view), not calibrated clinical doses.
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
