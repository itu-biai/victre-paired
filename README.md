# VICTRE-Paired

Code for **VICTRE-Paired v11**, a paired dataset for limited-angle digital breast tomosynthesis (DBT) reconstruction
built from the VICTRE in-silico imaging trial.

> Y. T. Basak, M. O. Unal, M. Ertas, I. Yildirim. *A multi-dose Monte Carlo digital breast tomosynthesis dataset
> with paired reference volumes.* Data Descriptor, submitted to Scientific Data (2026).

- **Data:** repository and DOI are given in the Data Descriptor.
- **Source:** VICTRE trial on The Cancer Imaging Archive (CC BY 3.0) and lesion locations from
  [DIDSR/VICTRE](https://github.com/DIDSR/VICTRE).

## What a record holds

2,993 phantoms (train 2,349 / validation 322 / test 322) in four breast-density classes. Each record holds the
25 Monte Carlo projections over ±25° and the mammogram at full dose; low-dose projections and mammograms at 50, 25
and 10 % dose, made by adding calibrated quantum and electronic noise in the pre-log intensity domain; an OS-SART
reconstruction of the full-dose projections (`sart_full`, the primary reference); VICTRE's own reconstruction
(`victre_fbp`, second reference, for the 2,760 phantoms whose reconstruction TCIA publishes); the breast mask, the
acquisition geometry, lesion locations and matched signal-absent controls. The Data Descriptor lists all 29 fields.

## Files

| file | purpose |
|---|---|
| `dataset_generate.py` | Downloads the VICTRE source data and writes every record (resumable). |
| `dataset_test.py` | Validates every record (keys, shapes, noise injection) and computes baseline metrics on the chosen splits; writes `validation.md` and `preview.png`. |
| `dbt_2026v1_manifest.csv` | Frozen splits of the 384-phantom development set; these phantoms keep their split. |
| `notebooks/` | Google Colab notebooks that generated the release and produced every figure, table and number of the Data Descriptor (below). |
| `notebooks/build/` | The scripts that write the notebooks (`python3 build_01_analysis.py 01_analysis_colab.ipynb`); edit these, not the notebooks. |

### Notebooks

| notebook | GPU | what it does |
|---|---|---|
| `00_generation_colab.ipynb` | yes | Runs `dataset_generate.py` over the collection in resumable sessions, then `dataset_test.py`, then writes `manifest.csv`. |
| `01_analysis_colab.ipynb` | yes | Registry of the release, DELTA check of every file, and per-phantom baselines, detection, depth alignment, noise and data fit on the test split. |
| `02_figures_tables_colab.ipynb` | no | Every figure, table and number of the Data Descriptor (`numbers.tex`), from the outputs of 01, 03–06. |
| `03_extra_analysis_colab.ipynb` | yes | Four classical methods × three doses × two projection sources (Monte Carlo and synthetic), noise spectra, forward-model residuals; SART pass count and SART-TV weight chosen on the validation split. |
| `04_training_colab.ipynb` | yes | A small supervised U-Net experiment showing that the training split is usable (supplementary material). |
| `05_diagnostics_colab.ipynb` | yes | Diagnostics of depth alignment, scale, noise ratio and zero voxels of the reference. |
| `06_delta_colab.ipynb` | yes | Checks the volume-height constant `DELTA` against VICTRE's own reconstruction. |
| `07_delta_patch_colab.ipynb` | yes | Recomputes `sart_full`, `breast_mask` and `volume_box_mm` of the dense, heterogeneous and scattered records with the corrected `DELTA`. |

The notebooks read the scripts and the data from Google Drive; the paths are set in their first cells.

## Usage

Both scripts need a CUDA GPU and the ASTRA toolbox. The dependencies are declared in each script header, so
[uv](https://docs.astral.sh/uv/) can run them directly:

```bash
uv run dataset_generate.py --cache ./victre_cache --output ./victre_paired_v11
uv run dataset_test.py --data ./victre_paired_v11
```

`--limit 8` runs a smoke test on the first eight phantoms. Noise is drawn from generators seeded by the phantom
identifier, so a rerun from the same source data gives the same low-dose arrays. The configuration of every run is
written to `config.json` in the output folder.

The v9 code that preceded this release is kept under the git tag `v9`.

## Licence

Code: MIT (see `LICENSE`). Data: see the Data Descriptor.
