#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "scipy",
#     "pydicom",
#     "torch",
#     "tqdm",
#     "astra-toolbox",
# ]
# ///
"""
Generate VICTRE-Paired v11 from the VICTRE in silico trial on TCIA.

For every eligible phantom, one file holds the 25 Monte Carlo DBT projections and the mammogram
of VICTRE's Siemens Mammomat Inspiration model at full dose; low-dose versions at 50, 25 and 10 %
dose, made by adding calibrated quantum and electronic noise in the pre-log intensity domain; an
OS-SART reconstruction of the full-dose projections (`sart_full`, the primary reference x_ref);
VICTRE's own reconstruction as a second reference (`victre_fbp`, where TCIA publishes it); and
lesion locations from DIDSR/VICTRE with matched signal-absent controls. Phantoms of the
development set (dbt_2026v1_manifest.csv) keep their fixed split.

The run is resumable: finished patients are skipped, failures are logged and retried
on the next run, and progress.json is rewritten after every patient.

Usage (needs a CUDA GPU for ASTRA):
    uv run dataset_generate.py --cache ./victre_cache --output ./victre_paired_v11
    uv run dataset_generate.py --limit 8            # smoke test on the first 8 phantoms
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import shutil
import tarfile
import time
import traceback
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import astra
import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from scipy import ndimage
from tqdm import tqdm

TCIA_API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
LOCATIONS_URL = "https://raw.githubusercontent.com/DIDSR/VICTRE/master/Locations/DBT-{density}.tar.gz"
DENSITIES = ("dense", "hetero", "scattered", "fatty")
DENSITY_SLICES = {"dense": 38, "hetero": 47, "scattered": 57, "fatty": 62}    # VICTRE recon depth, 1 mm slices


@dataclass(frozen=True)
class Config:
    seed: int = 42
    # dose levels: 50/25/10 per cent of VICTRE's own dose.
    # Keys become proj_50/25/10 and dm_50/25/10 because dose_key() uses round(100*f).
    dose_fractions: tuple = (0.50, 0.25, 0.10)
    min_counts: float = 0.5              # floor on simulated photon counts before the log
    # every eligible patient is used, so patients_per_group is not.
    # Split pattern repeated: 8 train, 1 val, 1 test.
    patients_per_group: int = 48         # unused, kept so config.json stays comparable across versions
    split_pattern: tuple = ("train",) * 8 + ("val", "test")
    # store VICTRE's own reconstruction as a second reference (NOT ground truth).
    # Present for the patients whose recon series TCIA publishes (all dense/hetero/scattered
    # plus a minority of fatty); flagged absent otherwise. Set False to drop the field.
    store_victre_fbp: bool = True
    # VICTRE DBT geometry (MC-GPU input files): source above the chest-wall edge
    num_views: int = 25
    half_angle_deg: float = 25.0
    sid_mm: float = 600.0
    sdd_mm: float = 650.0
    native_pixel_mm: float = 0.085
    downsample: int = 4
    dm_downsample: int = 2
    det_rows: int = 752                  # scan direction (3000 / 4, padded)
    det_cols: int = 384                  # chest wall to nipple (1500 / 4, padded)
    vol_rows: int = 408
    vol_cols: int = 336
    # preprocessing
    dicom_pedestal_dn: float = 400.0     # DN = 8 * (50 + 0.001469 * (Q - 5200)), Q in electrons
    dn_per_electron: float = 8 * 0.001469
    electronic_noise_electrons: float = 5200.0   # MC-GPU input: additive electronic noise per pixel
    exposed_min_dn: float = 800.0
    field_margin_px: int = 16
    breast_fraction: float = 0.85
    ring_px: tuple = (30, 120)
    mask_threshold: float = 0.08         # fraction of the reference's breast p99.5
    lesion_radius_mm: float = 2.5
    sart_iterations: int = 10
    sart_subsets: int = 5
    sart_depth_margin_slices: int = 8    # extra 1 mm slices above and below, cropped after SART


CFG = Config()
NATIVE_ROWS = {"dense": 1130, "hetero": 1148, "scattered": 1421, "fatty": 1624}
OFFX_C = -0.203
# DELTA: height of the bottom face of the volume above the detector plane, z0 = DELTA - (SDD - SID),
# in mm. 19.5 mm is VICTRE's analytic value: d_objbottom_det = 20.0 mm (FBP_DBTrecon.c) minus half
# of the 1 mm protective cover. An earlier pipeline used per-class values fitted to parallax
# (19.765 / 20.242 / 20.091 / 20.807 mm). Two lesion-free checks on the test phantoms fixed the
# value: the depth offset of sart_full against victre_fbp matched (DELTA - 19.5) slices in every
# class, and OS-SART rebuilt with DELTA from 19.0 to 20.5 mm agreed best with victre_fbp at
# 19.47-19.48 mm in all four classes. The released sart_full, breast_mask and volume_box_mm were
# recomputed with this value (notebook 07); the projections do not depend on it.
DELTA = {"dense": 19.500, "hetero": 19.500, "scattered": 19.500, "fatty": 19.500}


def dose_key(prefix: str, fraction: float) -> str:
    return f"{prefix}_{round(fraction * 100)}"


# ----------------------------------------------------------------------------- sources
def fetch(url: str, attempts: int = 4) -> bytes:
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=900) as response:
                return response.read()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(30 * (attempt + 1))
    raise RuntimeError(url)


def series_kind(description: str) -> str | None:
    if description == "Projection images":
        return "projections"
    if description == "DBT slices":
        return "recon"
    if description.startswith("mcgpu_image"):
        return "mammogram"
    return None


def tcia_index(cache: Path) -> dict[str, dict[str, dict]]:
    """PatientID -> series per kind, for patients that have DBT projections and a mammogram.

    VICTRE's own reconstruction is not needed: its depth is fixed per density (DENSITY_SLICES),
    and it is missing on TCIA for most fatty phantoms."""
    path = cache / "victre_series.json"
    if not path.exists():
        path.write_bytes(fetch(f"{TCIA_API}/getSeries?Collection=VICTRE&format=json"))
    patients: dict[str, dict[str, dict]] = defaultdict(dict)
    for series in json.loads(path.read_text()):
        kind = series_kind(series.get("SeriesDescription", ""))
        if kind:
            patients[series["PatientID"]][kind] = series
    return {pid: s for pid, s in patients.items() if "projections" in s and "mammogram" in s}


def lesion_index(cache: Path) -> dict[str, tuple[str, bool, np.ndarray]]:
    """PatientID -> (density, signal_present, rows of X Y Z lesion_number)."""
    index = {}
    for density in DENSITIES:
        path = cache / f"locations_{density}.tar.gz"
        if not path.exists():
            path.write_bytes(fetch(LOCATIONS_URL.format(density=density)))
        with tarfile.open(path) as archive:
            for member in archive.getmembers():
                name = Path(member.name).name
                if not (name.startswith("roi_S") and name.endswith(".loc")):
                    continue
                kind, seed = name[4:6], name[7:-4].lstrip("-")
                text = archive.extractfile(member).read().decode().splitlines()
                rows = np.array([[float(v) for v in line.split()] for line in text if len(line.split()) == 4],
                                np.float32).reshape(-1, 4)
                index[seed] = (density, kind == "SP", rows)
    return index


def download_series(series_uid: str, target: Path) -> Path:
    marker = target / ".complete"
    if not marker.exists():
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True)
        with zipfile.ZipFile(io.BytesIO(fetch(f"{TCIA_API}/getImage?SeriesInstanceUID={series_uid}"))) as archive:
            archive.extractall(target)
        marker.touch()
    return target


def read_sorted(folder: Path) -> np.ndarray:
    files = sorted(folder.rglob("*.dcm"), key=lambda f: int(pydicom.dcmread(f, stop_before_pixels=True).get("InstanceNumber") or 0))
    return np.stack([pydicom.dcmread(f).pixel_array.astype(np.float32) for f in files])


# ----------------------------------------------------------------------------- selection
# Frozen splits of the 384-phantom development set (BIAI Lab DBT 2026v1). Those phantoms keep
# their split, so a model trained on one dataset never sees the other's test phantoms.
FROZEN_SPLITS = Path(__file__).with_name("dbt_2026v1_manifest.csv")


def select_patients(series: dict, lesions: dict) -> list[dict]:
    """Every eligible phantom. Development-set phantoms keep their split and come first in each
    group; the others follow split_pattern (8 train, 1 val, 1 test, repeated) in random order.

    Eligible means: 25 DBT projections, a mammogram and DIDSR lesion locations. VICTRE's own
    reconstruction is NOT required -- but when TCIA publishes one its slice count must match
    the density, which also screens out the one source volume with a broken depth."""
    with open(FROZEN_SPLITS) as handle:
        frozen = {row["patient"]: row["split"] for row in csv.DictReader(handle)}
    rng = np.random.default_rng(CFG.seed)
    chosen = []
    for density in DENSITIES:
        pools = {True: [], False: []}
        for pid, entry in series.items():
            loc = lesions.get(pid.lstrip("-"))
            recon_slices = entry.get("recon", {}).get("ImageCount", DENSITY_SLICES[density])
            if loc and loc[0] == density and recon_slices == DENSITY_SLICES[density] and entry["projections"]["ImageCount"] == CFG.num_views:
                pools[loc[1]].append(pid)
        for present, pool in pools.items():
            order = [str(pid) for pid in rng.permutation(sorted(pool))]
            kept = [pid for pid in order if pid in frozen]
            for rank, pid in enumerate(kept + [pid for pid in order if pid not in frozen]):
                split = frozen[pid] if rank < len(kept) else CFG.split_pattern[(rank - len(kept)) % len(CFG.split_pattern)]
                chosen.append({"patient": pid, "density": density, "signal_present": present, "split": split, "rank": rank})
    missing = set(frozen) - {item["patient"] for item in chosen}
    if missing:
        raise RuntimeError(f"{len(missing)} development-set phantoms are no longer eligible")
    return sorted(chosen, key=lambda item: (item["rank"], DENSITIES.index(item["density"]), not item["signal_present"]))


# ----------------------------------------------------------------------------- projections
def pool2(image: np.ndarray, f: int) -> np.ndarray:
    return F.avg_pool2d(torch.from_numpy(np.ascontiguousarray(image, np.float32))[None, None], f)[0, 0].numpy()


def fit_to(stack: np.ndarray, rows: int, cols: int) -> np.ndarray:
    stack = stack[..., :rows, :cols]
    return np.pad(stack, [(0, 0)] * (stack.ndim - 2) + [(0, rows - stack.shape[-2]), (0, cols - stack.shape[-1])])


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    return ndimage.binary_fill_holes(labels == (np.argmax(sizes) + 1))


def local_variance(image: np.ndarray) -> np.ndarray:
    """Per-pixel noise variance from the 4-neighbour residual, averaged over 9x9 pixels."""
    axes = (-2, -1)
    neighbours = sum(np.roll(image, shift, axis) for shift in (1, -1) for axis in axes) / 4
    return ndimage.uniform_filter(((image - neighbours) ** 2 / 1.25), [1] * (image.ndim - 2) + [9, 9])


def view_to_projection(view: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Binned log attenuation, valid-ray mask and air level (DN above the pedestal).

    Flat field = median air signal 30-120 px around the breast; the 400 DN DICOM pedestal is
    removed; intensities are binned before the log; every collimator edge is excluded.
    """
    exposed = ndimage.binary_erosion(ndimage.binary_fill_holes(view > CFG.exposed_min_dn), iterations=CFG.field_margin_px)
    air_level = np.percentile(view[exposed], 90)
    breast = largest_component(exposed & (view < CFG.breast_fraction * air_level))
    distance = ndimage.distance_transform_edt(~breast)
    ring = (distance > CFG.ring_px[0]) & (distance < CFG.ring_px[1]) & exposed & (view > CFG.breast_fraction * air_level)
    signal = np.clip(view - CFG.dicom_pedestal_dn, 1.0, None)
    i0 = float(np.median(signal[ring]))
    binned = np.clip(pool2(signal, factor), 1.0, None)
    p = np.log(i0 / binned)
    footprint = pool2((distance <= 10).astype(np.float32), factor) > 0
    valid = pool2(exposed.astype(np.float32), factor) > 0.999
    p = np.where(footprint & valid, np.clip(p, 0, None), 0.0)
    return p.astype(np.float32), valid, i0


def effective_photons(p: np.ndarray, footprint: np.ndarray) -> float:
    """Photons per binned pixel before attenuation, from the noise already in the projections.

    Var(p) ~ 1 / (N0 exp(-p)); the local variance is taken from the 4-neighbour residual.
    Detector blur correlates neighbours, so this slightly overestimates N0.
    """
    inner = footprint & ndimage.binary_erosion(footprint, iterations=6, structure=np.ones((1, 3, 3)))
    estimates = 1.0 / (local_variance(p)[inner] * np.exp(-p[inner]))
    return float(np.median(estimates))


def electronic_sigma_photons(factor: int, i0_dn: float, n0: float) -> float:
    """Electronic noise std per binned pixel, in photons.

    MC-GPU adds 5200 electrons rms per native pixel, which is 61 DN through the DICOM gain.
    Binning averages factor^2 independent pixels, and one photon is i0 / N0 DN in the binned image."""
    return CFG.electronic_noise_electrons * CFG.dn_per_electron / factor * n0 / i0_dn


def reduce_dose(p: np.ndarray, footprint: np.ndarray, n0: float, sigma_e: float, fraction: float,
                rng: np.random.Generator) -> np.ndarray:
    """Scale-and-add noise in the pre-log domain.

    Full dose counts N = N0 exp(-p) carry variance N + sigma_e^2. Scaling by f and adding
    variance f(1-f)N + (1-f^2) sigma_e^2 gives the target f N + sigma_e^2 of a real f-dose scan.
    """
    counts = n0 * np.exp(-p.astype(np.float64))
    extra = fraction * (1 - fraction) * counts + (1 - fraction**2) * sigma_e**2
    low = fraction * counts + rng.standard_normal(p.shape) * np.sqrt(extra)
    low_p = np.log(fraction * n0 / np.clip(low, CFG.min_counts, None))
    return np.where(footprint, low_p, 0.0).astype(np.float32)


def noisy_versions(prefix: str, p: np.ndarray, footprint: np.ndarray, n0: float, sigma_e: float,
                   rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {dose_key(prefix, f): reduce_dose(p, footprint, n0, sigma_e, f, rng).astype(np.float16)
            for f in CFG.dose_fractions}


# ----------------------------------------------------------------------------- geometry and reconstruction
def volume_box(density: str, slices: int) -> tuple[float, float, float, float, float, float]:
    vox = CFG.native_pixel_mm * CFG.downsample
    chest_wall_y = -CFG.det_cols * vox / 2
    offx = (CFG.vol_rows * vox - NATIVE_ROWS[density] * CFG.native_pixel_mm) / 2.0 + OFFX_C
    offy = chest_wall_y + CFG.vol_cols * vox / 2
    offz = -(CFG.sdd_mm - CFG.sid_mm) + slices / 2.0 + DELTA[density]
    hx, hy, hz = CFG.vol_rows * vox / 2, CFG.vol_cols * vox / 2, slices / 2
    return offx - hx, offx + hx, offy - hy, offy + hy, offz - hz, offz + hz


def cone_vectors() -> np.ndarray:
    vox = CFG.native_pixel_mm * CFG.downsample
    angles = np.radians(np.linspace(-CFG.half_angle_deg, CFG.half_angle_deg, CFG.num_views))
    vectors = np.zeros((CFG.num_views, 12))
    vectors[:, 0] = CFG.sid_mm * np.sin(angles)
    vectors[:, 1] = -CFG.det_cols * vox / 2
    vectors[:, 2] = CFG.sid_mm * np.cos(angles)
    vectors[:, 5] = -(CFG.sdd_mm - CFG.sid_mm)
    vectors[:, 7] = vox
    vectors[:, 9] = vox
    return vectors


class Geometry:
    """ASTRA cone_vec geometry. Volumes (slices, rows, cols); projections (views, rows, cols)."""

    def __init__(self, density: str, slices: int):
        x0, x1, y0, y1, z0, z1 = volume_box(density, slices)
        m = CFG.sart_depth_margin_slices
        self.slices = slices
        self.vol_geom = astra.create_vol_geom(CFG.vol_cols, CFG.vol_rows, slices, x0, x1, y0, y1, z0, z1)
        self.vol_geom_extended = astra.create_vol_geom(CFG.vol_cols, CFG.vol_rows, slices + 2 * m, x0, x1, y0, y1, z0 - m, z1 + m)
        self.proj_geom = astra.create_proj_geom("cone_vec", CFG.det_rows, CFG.det_cols, cone_vectors())

    def os_sart(self, projections: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Ordered-subset SART: one ASTRA SIRT step per interleaved view subset, non-negative,
        invalid rays masked, on a volume extended in depth and cropped back to the grid.

        Limited-angle SIRT/SART smears tissue beyond the breast in depth; a volume that ends at
        the grid forces that smear into the visible slices (TUSEB findings, ASTRA section).
        """
        vectors = cone_vectors()
        volume_id = astra.data3d.create("-vol", self.vol_geom_extended, 0)
        subsets = []
        for start in range(CFG.sart_subsets):
            views = np.arange(start, CFG.num_views, CFG.sart_subsets)
            geom = astra.create_proj_geom("cone_vec", CFG.det_rows, CFG.det_cols, vectors[views])
            data = np.ascontiguousarray(projections[views].transpose(1, 0, 2), np.float32)
            mask = np.ascontiguousarray(valid[views].transpose(1, 0, 2), np.float32)
            subsets.append((astra.data3d.create("-proj3d", geom, data), astra.data3d.create("-proj3d", geom, mask)))
        for _ in range(CFG.sart_iterations):
            for sino_id, mask_id in subsets:
                config = astra.astra_dict("SIRT3D_CUDA")
                config["ProjectionDataId"], config["ReconstructionDataId"] = sino_id, volume_id
                config["option"] = {"MinConstraint": 0.0, "SinogramMaskId": mask_id}
                algorithm = astra.algorithm.create(config)
                astra.algorithm.run(algorithm, 1)
                astra.algorithm.delete(algorithm)
        volume = astra.data3d.get(volume_id)
        astra.data3d.delete([volume_id] + [i for pair in subsets for i in pair])
        m = CFG.sart_depth_margin_slices
        return np.ascontiguousarray(volume[m:m + self.slices].transpose(0, 2, 1))

    def backproject(self, projections: np.ndarray) -> np.ndarray:
        data = np.ascontiguousarray(projections.transpose(1, 0, 2), np.float32)
        vid, vol = astra.create_backprojection3d_gpu(data, self.proj_geom, self.vol_geom)
        astra.data3d.delete(vid)
        return np.ascontiguousarray(vol.transpose(0, 2, 1))

    def project(self, volume: np.ndarray) -> np.ndarray:
        sid, sino = astra.create_sino3d_gpu(np.ascontiguousarray(volume.transpose(0, 2, 1), np.float32), self.proj_geom, self.vol_geom)
        astra.data3d.delete(sid)
        return sino.transpose(1, 0, 2)

    def fbp(self, projections: np.ndarray) -> np.ndarray:
        """Cosine weighting from the chest-wall source, Hann-windowed ramp along the scan direction."""
        vox = CFG.native_pixel_mm * CFG.downsample
        u = (np.arange(CFG.det_rows) - CFG.det_rows / 2 + 0.5) * vox
        v = (np.arange(CFG.det_cols) + 0.5) * vox
        g = projections * (CFG.sdd_mm / np.sqrt(CFG.sdd_mm**2 + u[:, None] ** 2 + v[None, :] ** 2))[None].astype(np.float32)
        n = int(2 ** math.ceil(math.log2(CFG.det_rows * 1.6)))
        lo = (n - CFG.det_rows) // 2
        padded = np.pad(g, ((0, 0), (lo, n - CFG.det_rows - lo), (0, 0)), mode="edge")
        freq = np.fft.rfftfreq(n)
        ramp = 2.0 * freq * (0.5 + 0.5 * np.cos(np.pi * freq / freq.max()))
        filtered = np.fft.irfft(np.fft.rfft(padded, axis=1) * ramp[None, :, None], n=n, axis=1)[:, lo:lo + CFG.det_rows]
        return self.backproject(filtered.astype(np.float32))


# ----------------------------------------------------------------------------- labels
def breast_hull(geometry: Geometry, projections: np.ndarray, valid: np.ndarray) -> np.ndarray:
    outside = geometry.backproject(((projections <= 0) & valid).astype(np.float32))
    coverage = geometry.backproject(valid.astype(np.float32))
    return (outside <= 1e-3 * coverage.max()) & (coverage > 0.5 * coverage.max())


def breast_mask(ground_truth: np.ndarray, hull: np.ndarray) -> np.ndarray:
    """Projection hull (space carving) AND a smoothed reference x_ref above a fraction of its breast p99.5."""
    smooth = ndimage.uniform_filter(ground_truth, (1, 5, 5))
    level = np.percentile(smooth[hull], 99.5)
    return hull & (smooth > CFG.mask_threshold * level)


def lesion_table(rows: np.ndarray, signal_present: bool, shape: tuple[int, int, int]) -> np.ndarray:
    """(n, 5): slice, row, col, lesion_number, lesion_type (0 calcification cluster, 1 spiculated mass, -1 control)."""
    table = []
    for x, y, z, number in rows:
        r, c, s = x / CFG.downsample, y / CFG.downsample, z
        if 0 <= s < shape[0] and 0 <= r < shape[1] and 0 <= c < shape[2]:
            kind = (0 if number < 4 else 1) if signal_present else -1
            table.append([s, r, c, number, kind])
    return np.array(table, np.float32).reshape(-1, 5)


def lesion_roi(table: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Approximate label map: spheres of lesion_radius_mm (1 calcification cluster, 2 mass, 3 control)."""
    labels = np.zeros(shape, np.uint8)
    radius_px = CFG.lesion_radius_mm / (CFG.native_pixel_mm * CFG.downsample)
    zz, rr, cc = np.ogrid[: shape[0], : shape[1], : shape[2]]
    for s, r, c, _, kind in table:
        inside = ((rr - r) / radius_px) ** 2 + ((cc - c) / radius_px) ** 2 + ((zz - s) / CFG.lesion_radius_mm) ** 2 <= 1
        labels[inside] = {0: 1, 1: 2, -1: 3}[int(kind)]
    return labels


# ----------------------------------------------------------------------------- second reference
def load_victre_fbp(pid: str, entry: dict, density: str, cache: Path):
    """VICTRE's own reconstruction, resampled onto the voxel grid of sart_full.

    This is NOT the ground truth and must not be used as one: it is a smoothed filtered
    back projection with an undocumented filter on an arbitrary intensity scale, and
    `sart_full` remains the target. It is stored so that a result can also be reported against
    the trial's own reconstruction on the same phantoms, which otherwise needs a second dataset.

    Storage follows the source: uint16 normalised by the volume maximum, with the scale
    kept alongside. The arbitrary scale costs nothing because every scoring protocol in
    this release fits a gain and offset inside the breast mask first.

    Returns (volume uint16, scale, saturated fraction). Absent -> a zero-size array, which
    is why `victre_fbp_present` must be read before `victre_fbp`.

    Absent when: TCIA publishes no reconstruction series for the patient (true for most
    fatty phantoms), or its depth disagrees with the density, or the volume is degenerate.
    """
    empty = (np.zeros((0, 0, 0), np.uint16), np.float32(0.0), np.float32(0.0))
    series = entry.get("recon")
    if not CFG.store_victre_fbp or not series:
        return empty
    raw = read_sorted(download_series(series["SeriesInstanceUID"], cache / pid / "recon"))
    if raw.shape[0] != DENSITY_SLICES[density]:
        return empty
    # Some of the published volumes are clipped at the uint16 ceiling. That is the source's
    # own clipping and cannot be undone here; recording the fraction turns it into a number
    # the paper can state and a field analyses can filter on.
    saturated = float((raw >= 65535).mean())
    volume = fit_to(np.stack([pool2(sl, CFG.downsample) for sl in raw]), CFG.vol_rows, CFG.vol_cols)
    scale = float(volume.max())
    if not np.isfinite(scale) or scale <= 0:
        return empty
    stored = np.clip(np.round(volume / scale * 65535.0), 0, 65535).astype(np.uint16)
    return stored, np.float32(scale), np.float32(saturated)


# ----------------------------------------------------------------------------- patient
def build_patient(item: dict, series: dict, lesions: dict, cache: Path) -> dict:
    pid, density = item["patient"], item["density"]
    slices = DENSITY_SLICES[density]
    proj_dir = download_series(series[pid]["projections"]["SeriesInstanceUID"], cache / pid / "projections")
    dm_dir = download_series(series[pid]["mammogram"]["SeriesInstanceUID"], cache / pid / "mammogram")
    rng = np.random.default_rng(CFG.seed + int(pid.lstrip("-")) % 100_000)

    converted = [view_to_projection(v, CFG.downsample) for v in read_sorted(proj_dir)]
    proj_full = fit_to(np.stack([c[0] for c in converted]), CFG.det_rows, CFG.det_cols)
    valid = fit_to(np.stack([c[1] for c in converted]).astype(np.float32), CFG.det_rows, CFG.det_cols) > 0.5
    footprint = proj_full > 0
    n0 = effective_photons(proj_full, footprint)
    sigma_e = electronic_sigma_photons(CFG.downsample, float(np.median([c[2] for c in converted])), n0)
    projections = noisy_versions("proj", proj_full, footprint, n0, sigma_e, rng)

    dm, dm_valid, dm_i0 = view_to_projection(read_sorted(dm_dir)[0], CFG.dm_downsample)
    dm_footprint = dm > 0
    dm_n0 = effective_photons(dm[None], dm_footprint[None])
    dm_sigma_e = electronic_sigma_photons(CFG.dm_downsample, dm_i0, dm_n0)
    mammograms = noisy_versions("dm", dm, dm_footprint, dm_n0, dm_sigma_e, rng)

    geometry = Geometry(density, slices)
    sart_full = geometry.os_sart(proj_full, valid)
    victre_fbp, victre_scale, victre_saturated = load_victre_fbp(pid, series[pid], density, cache)
    mask = breast_mask(sart_full, breast_hull(geometry, proj_full, valid))
    shape = (slices, CFG.vol_rows, CFG.vol_cols)
    table = lesion_table(lesions[pid.lstrip("-")][2], item["signal_present"], shape)
    return {
        "proj_full": proj_full.astype(np.float16), **projections, "proj_valid": valid,
        "sart_full": sart_full.astype(np.float32), "breast_mask": mask,
        # second reference; read victre_fbp_present before victre_fbp
        "victre_fbp": victre_fbp, "victre_fbp_scale": victre_scale,
        "victre_fbp_present": bool(victre_fbp.size > 0),
        "victre_fbp_saturated": victre_saturated,
        "dm_full": dm.astype(np.float16), **mammograms, "dm_valid": dm_valid,
        "lesions": table, "lesion_roi": lesion_roi(table, shape),
        "geometry_vectors": cone_vectors().astype(np.float32),
        "volume_box_mm": np.array(volume_box(density, slices), np.float32),
        "voxel_mm": np.array([1.0, CFG.native_pixel_mm * CFG.downsample, CFG.native_pixel_mm * CFG.downsample], np.float32),
        "dose_fractions": np.array(CFG.dose_fractions, np.float32),
        "photons_per_pixel": np.float32(n0), "electronic_sigma": np.float32(sigma_e),
        "dm_photons_per_pixel": np.float32(dm_n0), "dm_electronic_sigma": np.float32(dm_sigma_e),
        "patient": pid, "density": density, "signal_present": item["signal_present"],
    }


def write_progress(output: Path, items: list[dict], failed: dict[str, str], started: float, done_now: int) -> None:
    done = [i for i in items if (output / i["split"] / f"{i['patient']}.npz").exists()]
    elapsed = time.time() - started
    rate = elapsed / done_now if done_now else None
    progress = {
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"), "total": len(items), "done": len(done),
        "failed": failed, "elapsed_hours": round(elapsed / 3600, 2),
        "minutes_per_patient": round(rate / 60, 2) if rate else None,
        "eta_hours": round(rate * (len(items) - len(done)) / 3600, 1) if rate else None,
        "size_gb": round(sum(f.stat().st_size for f in output.rglob("*.npz")) / 1e9, 2),
    }
    (output / "progress.json").write_text(json.dumps(progress, indent=2))


def write_manifest(output: Path, items: list[dict]) -> None:
    rows = []
    for item in items:
        target = output / item["split"] / f"{item['patient']}.npz"
        if target.exists():
            with np.load(target) as data:
                rows.append({**{k: item[k] for k in ("patient", "density", "signal_present", "split")},
                             "slices": int(data["sart_full"].shape[0]),
                             "lesions": int((data["lesions"][:, 4] >= 0).sum()),
                             "photons_per_pixel": round(float(data["photons_per_pixel"]), 1),
                             "electronic_sigma": round(float(data["electronic_sigma"]), 2),
                             # which patients carry the second reference, and how much
                             # of the source volume sits at its uint16 ceiling
                             "victre_fbp": int(bool(data["victre_fbp_present"])),
                             "victre_fbp_saturated": round(float(data["victre_fbp_saturated"]), 6),
                             "size_mb": round(target.stat().st_size / 1e6, 1)})
    if rows:
        with open(output / "manifest.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"{len(rows)} patients, {sum(r['slices'] for r in rows)} slices")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("victre_cache"))
    parser.add_argument("--output", type=Path, default=Path("victre_paired_v11"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--keep-cache", action="store_true", help="keep downloaded DICOM after a patient is written")
    # Colab plumbing. None of these touch what is written, only where the run looks
    # for finished work and how often it reports. On a machine with a local disk, omit them.
    parser.add_argument("--done-list", type=Path, default=None,
                        help="file with one finished patient id per line; those are skipped "
                             "without touching the output directory (use when finished files "
                             "live on slow storage such as a mounted Drive)")
    parser.add_argument("--max-new", type=int, default=0,
                        help="stop after this many newly written patients, so the caller can "
                             "move them to durable storage and run again")
    parser.add_argument("--progress-every", type=int, default=1,
                        help="write progress.json every N patients; raise it when the output "
                             "directory is on slow storage")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)

    series, lesions = tcia_index(args.cache), lesion_index(args.cache)
    items = select_patients(series, lesions)
    if args.limit:
        items = items[: args.limit]
    (args.output / "config.json").write_text(json.dumps(asdict(CFG), indent=2))
    # patients finished in an earlier session, listed instead of looked up on disk
    elsewhere = set()
    if args.done_list and args.done_list.exists():
        elsewhere = {ln.strip() for ln in args.done_list.read_text().splitlines() if ln.strip()}
        print(f"{len(elsewhere)} patients already finished elsewhere, skipping")
    failed: dict[str, str] = {}
    started, done_now = time.time(), 0
    for item in tqdm(items, desc="patients"):
        if item["patient"] in elsewhere:
            continue
        target = args.output / item["split"] / f"{item['patient']}.npz"
        if target.exists():
            continue
        if args.max_new and done_now >= args.max_new:
            print(f"--max-new {args.max_new} reached; rerun to continue")
            break
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            contents = build_patient(item, series, lesions, args.cache)
            partial = target.with_suffix(".partial.npz")
            np.savez_compressed(partial, **contents)
            partial.rename(target)
            done_now += 1
            if not args.keep_cache:
                shutil.rmtree(args.cache / item["patient"], ignore_errors=True)
        except Exception as error:
            failed[item["patient"]] = f"{type(error).__name__}: {error}"
            print(f"FAILED {item['patient']}:\n{traceback.format_exc()}", flush=True)
        if done_now % max(args.progress_every, 1) == 0 or failed:
            write_progress(args.output, items, failed, started, done_now)
    write_progress(args.output, items, failed, started, done_now)
    write_manifest(args.output, [i for i in items if i["patient"] not in elsewhere])


if __name__ == "__main__":
    main()
