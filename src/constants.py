"""
Shared constants for the VICTRE-PAIRED v6 pipeline.

All acquisition geometry, dose, alignment and split parameters used by the
generation, validation and baseline scripts are defined here so that a single
edit propagates everywhere and the released dataset is reproducible bit-for-bit
given the same VICTRE source data.

Geometry corresponds to the LEAP (LivermorE AI Projector) modular-beam model
after 4x spatial downsampling of the VICTRE projections and reconstructions.
"""

# ── Acquisition geometry (LEAP modular-beam, post-downsampling) ────────────
SID = 630.0          # source-to-isocenter distance (mm)
SDD = 650.0          # source-to-detector distance (mm)
DET_PIX = 0.34       # detector pixel pitch (mm)
NA = 25              # number of Monte-Carlo projection views
ANGLE_DEG = 25.0     # half angular range; views span [-25 deg, +25 deg]

PROJ_ROWS = 752      # projection rows  (after downsampling)
PROJ_COLS = 384      # projection cols  (after downsampling)
RECON_Z = 56         # reconstruction slices (isotropic z-resample target)
RECON_H = 408        # reconstruction height
RECON_W = 336        # reconstruction width
VOX_XY = 0.34        # in-plane voxel size (mm)
VOX_Z = 1.0          # slice thickness (mm)

DS = 4               # spatial downsampling factor applied to VICTRE data

# ── Dose / noise model ─────────────────────────────────────────────────────
# Poisson gain per dose level; larger gain = fewer effective photons = noisier.
DOSES = {"full": 0.001, "half": 0.002, "quarter": 0.004}
S_ELEC = 0.010       # additive Gaussian electronic-noise std
# `noisy_proj` (the default noisy projection array) is generated at HALF dose.

# ── Density-dependent projection alignment offsets (empirical) ─────────────
# (slope, intercept, column-shift): per-view row shift = slope*view + intercept,
# with a constant column shift. Derived by maximizing agreement between the
# real projections and the synthetic forward projection of the reconstruction.
OFFSET = {
    "scattered": (3.60, -70, 16),
    "hetero":    (2.80, -96, 16),
    "fatty":     (3.30, -40, 17),
    "dense":     (2.10, -90, 16),
}

# ── Penumbra strip removal ────────────────────────────────────────────────
VALLEY = 3.22        # attenuation threshold marking the penumbra valley
DILATE = 1           # binary-dilation iterations around the flagged strip

# ── Train / val / test split (deterministic) ──────────────────────────────
SEED_SPLIT = 42
SPLIT = {"train": 0.80, "val": 0.10, "test": 0.10}
CHUNK = 8            # patients per .npz chunk

# ── Reference metadata ─────────────────────────────────────────────────────
# VICTRE reconstruction native slice count per breast density (before z-resample).
NZMAP = {"dense": 38, "hetero": 47, "scattered": 57, "fatty": 62}

# Expected split sizes for the released dataset (chunks, patients).
EXPECT = {"train": (276, 2208), "val": (35, 276), "test": (35, 277)}
N_PATIENTS_TOTAL = 2761

# ── Known corrupt patient (VICTRE-sourced; documented, not removed) ────────
# SEED 208084664 has a degenerate reconstruction (5 native slices, no depth
# information). Kept in the release for completeness; exclude in evaluation.
BROKEN_SEEDS = {208084664}


def fmt_seconds(s):
    """Human-readable H/M/S from a duration in seconds."""
    s = int(s)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"
