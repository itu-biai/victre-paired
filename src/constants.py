"""
VICTRE-Paired — shared constants.

Single source of truth for geometry, normalization, dose and split parameters.
All values below are the ones actually used to produce the released dataset and
are verified against every chunk by validate_dataset.py.
"""

# ---------------------------------------------------------------------------
# Reconstruction volume (per patient, after downsampling)
# ---------------------------------------------------------------------------
ZOUT   = 56          # output slices (all patients resampled to 56)
TH     = 408         # volume rows
TW     = 336         # volume cols
VOX_XY = 0.34        # in-plane voxel size [mm]  (= detector pixel after 4x)

# ---------------------------------------------------------------------------
# Projection geometry (constant across the whole dataset)
# ---------------------------------------------------------------------------
NA      = 25         # number of views
PH      = 752        # projection rows
PW      = 384        # projection cols
DET_PIX = 0.34       # detector pixel size [mm] (after 4x downsampling)
ANG     = 25.0       # half angular range [deg]  → views span [-25, +25]
SID     = 600.0      # source-to-isocenter distance [mm]
SDD     = 650.0      # source-to-detector distance [mm]

# Native detector / reconstruction pixel of the VICTRE source (before 4x)
NATIVE_PIX = 0.085   # [mm]

# ---------------------------------------------------------------------------
# Per-density native reconstruction dimensions (class-constant).
# native_x = long axis (rows), native_y = short axis (cols), native_z = slices.
# ---------------------------------------------------------------------------
NATIVE_XY = {
    "dense":     (1130, 477),
    "hetero":    (1148, 753),
    "scattered": (1421, 1024),
    "fatty":     (1624, 1324),
}

# ---------------------------------------------------------------------------
# Analytic volume placement for the LEAP modular-beam geometry.
#
#   geom_vox_z = native_z / ZOUT
#   geom_offx  = (TH*VOX_XY - native_x*NATIVE_PIX)/2 + OFFX_C
#   geom_offy  = OFFY_A + OFFY_B * native_y
#   geom_offz  = -(SDD - SID) + ZOUT*geom_vox_z/2 + DELTA[density]
#
# These are stored per patient as geom_* fields; the constants below let you
# rederive them and are the ones validate_dataset.py checks against.
# ---------------------------------------------------------------------------
OFFX_C = -0.203
OFFY_A = -5.142
OFFY_B = 0.0014214

# Per-density z-offset term. Tuned so that the residual parallax between
# A(clean) and clean_proj is minimized on a held-out probe set.
DELTA = {
    "dense":     19.765,
    "hetero":    20.242,
    "scattered": 20.091,
    "fatty":     20.807,
}

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
P_RECON  = 99.5      # clean is normalized to this percentile = 1.0
P_PROJ   = 99.8      # clean_proj is normalized to this percentile = 1.0
MASK_THR = 0.08      # breast mask = clean > MASK_THR

# ---------------------------------------------------------------------------
# Noise / dose model (intensity domain)
#
#   I0 = 1/gain                              photon budget per pixel per view
#   I  = I0 * exp(-clip(p,0,1) * proj_scale) transmitted photons
#   N  ~ Poisson(I) + Normal(0, S_ELEC*I0*0.02)
#   p_noisy = clip(-log(N/I0)/proj_scale, 0, 1)
#
# noise_seed = seed ; the rng is default_rng(noise_seed*10 + dose_idx).
# dose_idx: 0 = full, 1 = half, 2 = quarter.
# ---------------------------------------------------------------------------
DOSES  = {"full": 0.001, "half": 0.002, "quarter": 0.004}   # gain per dose
DOSE_IDX = {"full": 0, "half": 1, "quarter": 2}
S_ELEC = 0.010       # electronic noise scale

# ---------------------------------------------------------------------------
# Split (read from the VICTRE source assignment, not re-shuffled)
# ---------------------------------------------------------------------------
SEED_SPLIT = 42

# ---------------------------------------------------------------------------
# Patients excluded from evaluation (kept in the dataset for completeness)
# ---------------------------------------------------------------------------
BROKEN_SEEDS = {208084664}   # degenerate VICTRE reconstruction

DENSITIES = ["dense", "hetero", "scattered", "fatty"]
