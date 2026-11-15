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
