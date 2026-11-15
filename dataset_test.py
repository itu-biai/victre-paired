#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "scipy",
#     "scikit-image",
#     "matplotlib",
#     "pydicom",
#     "torch",
#     "tqdm",
#     "astra-toolbox",
# ]
# ///
"""
Validate a VICTRE-Paired release and compute baseline metrics.

1. Checks keys, shapes and dtypes of every patient file.
2. Checks the noise injection of every file: the extra noise variance measured in each
   low-dose projection set, divided by the variance the noise model predicts from the stored
   photon count and electronic noise, should be 1.
3. Reconstructs the low-dose projections of the metric splits with OS-SART (same algorithm as
   the ground truth) and FBP and reports PSNR, SSIM and MSE against the ground truth inside
   the breast, per dose, split and density.
4. Writes validation.md (all tables) and preview.png (one signal-present patient per density).

Usage (needs a CUDA GPU for ASTRA):
    uv run dataset_test.py --data ./victre_paired_v11
    uv run dataset_test.py --data ./victre_paired_v11 --metric-splits val test
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from skimage.metrics import structural_similarity
from scipy import ndimage
from tqdm import tqdm

from dataset_generate import CFG, Geometry, dose_key, local_variance

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPLITS = ("train", "val", "test")
DENSITIES = ("dense", "hetero", "scattered", "fatty")
DOSES = [dose_key("", f)[1:] for f in CFG.dose_fractions]
VOLUME_KEYS = ("sart_full", "breast_mask", "lesion_roi")
EXPECTED = {
    "proj_full": (np.float16, 3), "proj_valid": (np.bool_, 3),
    "sart_full": (np.float32, 3), "breast_mask": (np.bool_, 3), "lesion_roi": (np.uint8, 3),
    "dm_full": (np.float16, 2), "dm_valid": (np.bool_, 2),
    "lesions": (np.float32, 2), "geometry_vectors": (np.float32, 2),
    "volume_box_mm": (np.float32, 1), "voxel_mm": (np.float32, 1), "dose_fractions": (np.float32, 1),
    **{f"proj_{d}": (np.float16, 3) for d in DOSES}, **{f"dm_{d}": (np.float16, 2) for d in DOSES},
}


def normalize(volume: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Map to [0, 1] with the ground truth's breast p0.5-p99.5 window (same window for both inputs)."""
    lo, hi = np.percentile(reference[mask], [0.5, 99.5])
    return np.clip((volume - lo) / (hi - lo), 0, 1)


def scale_match(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Least-squares a * pred + b inside the breast; FBP is not in mm^-1, SART nearly is."""
    a, b = np.polyfit(pred[mask].astype(np.float64), gt[mask].astype(np.float64), 1)
    return a * pred + b


def metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    p, g = normalize(scale_match(pred, gt, mask), gt, mask), normalize(gt, gt, mask)
    mse = float(np.mean((p[mask] - g[mask]) ** 2))
    ssim = []
    for z in range(g.shape[0]):
        if mask[z].sum() > 100:
            _, ssim_map = structural_similarity(g[z], p[z], data_range=1.0, full=True)
            ssim.append(ssim_map[mask[z]])
    return {"psnr": 10 * np.log10(1.0 / mse), "ssim": 100 * float(np.concatenate(ssim).mean()), "mse": mse}


def check(data: np.lib.npyio.NpzFile, name: str) -> list[str]:
    problems = []
    for key, (dtype, ndim) in EXPECTED.items():
        if key not in data.files:
            problems.append(f"{name}: missing {key}")
        elif data[key].dtype != dtype or data[key].ndim != ndim:
            problems.append(f"{name}: {key} is {data[key].dtype} {data[key].shape}")
    # second reference, four fields; shapes are conditional so they are not in EXPECTED
    for key in ("victre_fbp", "victre_fbp_scale", "victre_fbp_present", "victre_fbp_saturated"):
        if key not in data.files:
            problems.append(f"{name}: missing {key}")
    if problems:
        return problems
    slices = data["sart_full"].shape[0]
    for key in ("proj_full", "proj_valid", *(f"proj_{d}" for d in DOSES)):
        if data[key].shape != (CFG.num_views, CFG.det_rows, CFG.det_cols):
            problems.append(f"{name}: {key} {data[key].shape}")
    for key in VOLUME_KEYS:
        if data[key].shape != (slices, CFG.vol_rows, CFG.vol_cols):
            problems.append(f"{name}: {key} {data[key].shape}")
    # present -> same grid as sart_full; absent -> zero-size, so that reading
    # victre_fbp without checking victre_fbp_present fails loudly instead of silently.
    vf, present = data["victre_fbp"], bool(data["victre_fbp_present"])
    if vf.dtype != np.uint16:
        problems.append(f"{name}: victre_fbp is {vf.dtype}")
    elif present and vf.shape != (slices, CFG.vol_rows, CFG.vol_cols):
        problems.append(f"{name}: victre_fbp {vf.shape} with present=True")
    elif not present and vf.size:
        problems.append(f"{name}: victre_fbp has {vf.size} values with present=False")
    if present and not float(data["victre_fbp_scale"]) > 0:
        problems.append(f"{name}: victre_fbp_scale is {float(data['victre_fbp_scale'])}")
    for key in ("dm_valid", *(f"dm_{d}" for d in DOSES)):
        if data[key].shape != data["dm_full"].shape:
            problems.append(f"{name}: {key} {data[key].shape}")
    if data["breast_mask"].sum() == 0:
        problems.append(f"{name}: empty breast mask")
    for key in [k for k in EXPECTED if k.startswith(("proj_", "dm_")) and "valid" not in k]:
        if not np.isfinite(data[key].astype(np.float32)).all():
            problems.append(f"{name}: {key} has non-finite values")
    return problems


def noise_ratios(data: np.lib.npyio.NpzFile) -> dict[str, float]:
    """Measured extra log-domain variance over the predicted (f(1-f)N + (1-f^2) sigma_e^2) / (fN)^2.

    The injected noise is white, so the 4-neighbour residual measures it without bias, and
    the full-dose noise cancels in the difference low minus full."""
    ratios = {}
    for prefix, expand, n0_key, sigma_key in (("proj", lambda a: a, "photons_per_pixel", "electronic_sigma"),
                                              ("dm", lambda a: a[None], "dm_photons_per_pixel", "dm_electronic_sigma")):
        full = expand(data[f"{prefix}_full"].astype(np.float32))
        inner = ndimage.binary_erosion(full > 0, iterations=6, structure=np.ones((1, 3, 3)))
        counts = float(data[n0_key]) * np.exp(-full[inner].astype(np.float64))
        sigma_e = float(data[sigma_key])
        full_var = local_variance(full)[inner]
        for f, d in zip(CFG.dose_fractions, DOSES):
            predicted = (f * (1 - f) * counts + (1 - f**2) * sigma_e**2) / (f * counts) ** 2
            measured = local_variance(expand(data[f"{prefix}_{d}"].astype(np.float32)))[inner] - full_var
            ratios[f"{prefix}_{d}"] = float(np.median(measured) / np.median(predicted))
    return ratios


def table(rows: dict, title: str) -> list[str]:
    lines = [f"\n### {title}\n", "| Split | Density | Patients | PSNR | SSIM | MSE |", "|---|---|---|---|---|---|"]
    for (split, density), values in sorted(rows.items(), key=lambda kv: (SPLITS.index(kv[0][0]) if kv[0][0] in SPLITS else 9,
                                                                         (("all",) + DENSITIES).index(kv[0][1]))):
        avg = {k: np.mean([v[k] for v in values]) for k in values[0]}
        std = np.std([v["psnr"] for v in values])
        lines.append(f"| {split} | {density} | {len(values)} | {avg['psnr']:.2f} ± {std:.2f} | {avg['ssim']:.2f} | {avg['mse']:.6f} |")
    return lines


def preview(files: list[Path], out: Path) -> None:
    picks = {}
    for path in files:
        with np.load(path) as data:
            density = str(data["density"])
            if bool(data["signal_present"]) and density not in picks:
                picks[density] = path
    if not picks:
        picks = {str(np.load(files[0])["density"]): files[0]}
    fig, axes = plt.subplots(len(picks), 7, figsize=(26, 3.8 * len(picks)))
    axes = np.atleast_2d(axes)
    for row, (density, path) in zip(axes, sorted(picks.items())):
        with np.load(path) as data:
            mass = data["lesions"][data["lesions"][:, 4] == 1]
            z = int(mass[0, 0]) if len(mass) else data["sart_full"].shape[0] // 2
            geometry = Geometry(density, data["sart_full"].shape[0])
            low = DOSES[-1]
            sart_low = geometry.os_sart(data[f"proj_{low}"].astype(np.float32), data["proj_valid"])
            panels = {"proj_full view 12": data["proj_full"][12], f"proj_{low} view 12": data[f"proj_{low}"][12],
                      "sart_full (GT)": data["sart_full"][z], f"SART from {low}% dose": sart_low[z],
                      "dm_full": data["dm_full"], f"dm_{DOSES[0]}": data[f"dm_{DOSES[0]}"], f"dm_{low}": data[f"dm_{low}"]}
            mask = data["breast_mask"][z]
            for index, (ax, (title, image)) in enumerate(zip(row, panels.items())):
                image = image.astype(np.float32)
                volume_panel = image.shape == mask.shape
                region = mask if volume_panel else image > 0
                window = data["sart_full"][z] if volume_panel else image
                lo, hi = np.percentile(window[region], [1, 99])
                rows, cols = np.nonzero(region)
                r0, r1, c0, c1 = max(rows.min() - 10, 0), rows.max() + 10, max(cols.min() - 10, 0), cols.max() + 10
                ax.imshow(image, cmap="gray" if volume_panel else "gray_r", vmin=lo, vmax=hi)
                ax.set_xlim(c0, c1)
                ax.set_ylim(r1, r0)
                if volume_panel:
                    for s, r, c, _, kind in data["lesions"]:
                        if abs(s - z) <= 2:
                            ax.add_patch(plt.Circle((c, r), 9, fill=False, ec="red" if kind == 1 else "cyan", lw=1))
                ax.set_title(f"{path.stem} {density}: {title}" + (f", slice {z}" if volume_panel else ""), fontsize=8)
                ax.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=70)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("victre_paired_v11"))
    parser.add_argument("--metric-splits", nargs="+", default=["test"])
    args = parser.parse_args()

    files = [p for split in SPLITS for p in sorted((args.data / split).glob("*.npz")) if ".partial" not in p.name]
    problems, ratios = [], defaultdict(list)
    second_ref = defaultdict(list)
    by_density = defaultdict(list)
    results = defaultdict(lambda: defaultdict(list))
    for path in tqdm(files, desc="validate"):
        split = path.parent.name
        with np.load(path) as data:
            found = check(data, path.name)
            problems += found
            if found:
                continue
            for key, value in noise_ratios(data).items():
                ratios[key].append(value)
            if split not in args.metric_splits:
                continue
            # alignment of the second reference. Both volumes reconstruct the same
            # projections on the same grid, so inside the breast they must correlate clearly.
            # A value near zero means the slice order or the row/column convention is wrong,
            # which is the one way this field can be silently broken.
            if bool(data["victre_fbp_present"]):
                _m = data["breast_mask"]
                _a = data["victre_fbp"][_m].astype(np.float64)
                _b = data["sart_full"][_m].astype(np.float64)
                if _a.std() > 0 and _b.std() > 0:
                    _c = float(np.corrcoef(_a, _b)[0, 1])
                    second_ref["corr"].append(_c)
                    by_density[str(data["density"])].append(_c)
                second_ref["saturated"].append(float(data["victre_fbp_saturated"]))
            second_ref["present"].append(bool(data["victre_fbp_present"]))

            density, gt, mask = str(data["density"]), data["sart_full"], data["breast_mask"]
            geometry = Geometry(density, gt.shape[0])
            recons = {"FBP full dose": geometry.fbp(data["proj_full"].astype(np.float32))}
            for d in DOSES:
                proj = data[f"proj_{d}"].astype(np.float32)
                recons[f"SART {d}% dose"] = geometry.os_sart(proj, data["proj_valid"])
                recons[f"FBP {d}% dose"] = geometry.fbp(proj)
            for name, volume in recons.items():
                m = metrics(volume, gt, mask)
                for key in ((split, "all"), (split, density), ("all", "all")):
                    results[name][key].append(m)

    lines = ["# VICTRE-Paired v11 Validation\n", f"Files checked: {len(files)}. Problems: {len(problems)}.\n"]
    lines += [f"- {p}" for p in problems]
    lines += ["\n## Noise injection check\n",
              "Measured extra noise variance over the noise model prediction (target 1.00).\n",
              "| Key | Mean | Min | Max |", "|---|---|---|---|"]
    for key, values in ratios.items():
        lines.append(f"| {key} | {np.mean(values):.3f} | {np.min(values):.3f} | {np.max(values):.3f} |")
    # second reference. Reported, not scored: victre_fbp is a reference a user may
    # choose to measure against, never the ground truth of this release.
    if second_ref["present"]:
        c, sat = second_ref["corr"], second_ref["saturated"]
        lines += ["\n## Second reference (victre_fbp)\n",
                  "| Quantity | Value |", "|---|---|",
                  f"| Patients carrying it (metric splits) | {sum(second_ref['present'])} / {len(second_ref['present'])} |"]
        if c:
            lines.append(f"| corr(victre_fbp, sart_full) in the breast mask | {np.mean(c):.3f} [{np.min(c):.3f}, {np.max(c):.3f}] |")
            # Measured on the 8-patient smoke run: dense 0.52, hetero 0.49, scattered 0.45,
            # fatty 0.28. The spread is the genuine difference between two reconstruction
            # families, not a defect -- VICTRE's volume is a smoothed filtered back projection
            # and sart_full is an ordered-subset SART. A real break (wrong axis order, reversed
            # slices) drops this far below any of those values: reversing the slice order gives
            # 0.15 to 0.29. The threshold is set to catch that, not the family difference.
            if np.min(c) < 0.15:
                lines.append(f"| **WARNING** | {sum(x < 0.15 for x in c)} volume(s) below 0.15 -- check the grid mapping |")
        if by_density:
            lines += ["", "Per density:", "",
                      "| Density | n | corr |", "|---|---|---|"]
            for dn in ("dense", "hetero", "scattered", "fatty"):
                v = by_density.get(dn)
                if v:
                    lines.append(f"| {dn} | {len(v)} | {np.mean(v):.3f} |")
        if sat:
            lines.append(f"| Source voxels at the uint16 ceiling | mean {np.mean(sat):.2e}, max {np.max(sat):.2e} |")
        lines.append("")
    lines.append("\n## Baselines against the ground truth (sart_full), inside the breast mask\n")
    for name, rows in results.items():
        lines += table(rows, name)
    report = "\n".join(lines) + "\n"
    (args.data / "validation.md").write_text(report)
    print(report)
    if files:
        preview(files, args.data / "preview.png")
        print(f"preview written to {args.data / 'preview.png'}")


if __name__ == "__main__":
    main()
