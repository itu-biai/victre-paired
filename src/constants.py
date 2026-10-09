"""
VICTRE-Paired — shared constants (data format v9).

Single source of truth for geometry, normalisation, photon-budget and split
parameters. Every value here is the one actually used to produce the released
dataset and is verified against every chunk by validate_dataset.py.

What changed in v9 and why: see CHANGELOG_v9.md.
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
# Normalisation (v9)
#
# Both arrays are divided by their own maximum and NOTHING is clipped, so
# exactly one sample per patient sits at the uint16 ceiling by construction:
#
#   clean      = t / max(t)                 t = resampled reconstruction volume
#   clean_proj = p / max(p)                 p = flat-field-corrected attenuation
#
# v8 divided by a percentile and clipped to [0, 1], which flattened the bright
# tail: 0.68 % of breast voxels and 0.2 % of nonzero projection pixels sat at
# the ceiling. The two percentiles below are kept only to document that, and to
# let validate_dataset.py report max/percentile ratios.
# ---------------------------------------------------------------------------
FORMAT_VERSION = "v9"
NORM_ANCHOR    = "max"     # v9; v8 was "percentile"
P_RECON_V8     = 99.5      # what v8 normalised clean to
P_PROJ_V8      = 99.8      # what v8 normalised clean_proj to

# Breast mask.
#
# CAREFUL: the threshold is applied to the NORMALISED volume, so its physical
# level depends on the normalisation anchor. v8's divisor was p99.5, v9's is the
# maximum, and max/p99.5 is about 1.77 (worst 2.01) — so a literal 0.08 means a
# ~1.77x higher physical level in v9 than in v8, by a patient-dependent factor.
#
# MASK_ANCHOR decides which one is used:
#   "max"   -> mask = clean > MASK_THR              (as generated; v9 literal)
#   "p99.5" -> mask = clean > MASK_THR * p99.5/max  (same physical level as v8,
#                                                    i.e. scale-free)
# Whether this matters at all depends on how wide the air/tissue gap is; the
# measurement is G1 STAGE 0 (Dice between the v8 and v9 masks of the same
# patient). Until that says otherwise the generated value stands.
MASK_THR    = 0.08
# MEASURED (29 Sep): with "max" the mask lost 14.3 % of its volume on average and
# 20.4 % in the worst case, concentrated in dense breasts (Dice against the
# previous format 0.923 mean, 0.886 worst). The lost region is anatomically
# structured -- a band along the chest wall plus the skin rim -- not threshold
# noise. "p99.5" puts the threshold back at the physical level the previous
# format used, so the breast support is identical and the primary metric's
# support does not move for a reason unrelated to the projections.
MASK_ANCHOR = "p99.5"

# ---------------------------------------------------------------------------
# Noise model (unchanged from v8) -- see noise.py for the exact formula.
#
#   I0 = 1/gain                       photons per 0.34 mm pixel per view
#   I  = I0 * exp(-p * proj_scale)    p = clean_proj / 65535
#   N  ~ Poisson(I) + Normal(0, ELEC_PHOTONS)
#   p_noisy = -log(max(N, PHOTON_FLOOR)/I0) / proj_scale   (inside the illuminated box)
#
# Stored as uint16 with p_noisy = (u16 - NOISY_ZERO) * NOISY_STEP, i.e. the
# range [NOISY_LO, NOISY_HI]; nothing is clipped to [0,1]. Generated from the
# released clean_proj, so noisy_proj* regenerate bit-for-bit from the record.
# RNG: default_rng(seed*10 + dose_idx), dose_idx full:0 half:1 quarter:2.
# The three budgets are relative noise levels, not calibrated clinical doses
# (I0 = 1000 is roughly two orders of magnitude below a clinical air-side count).
# ---------------------------------------------------------------------------
DOSES    = {"full": 0.001, "half": 0.002, "quarter": 0.004}   # gain per budget
DOSE_IDX = {"full": 0, "half": 1, "quarter": 2}
ELEC_PHOTONS = 2.0       # Gaussian electronic noise, std in photons
PHOTON_FLOOR = 0.5       # photons; floor before the log
NOISY_LO     = -0.25     # stored range [NOISY_LO, NOISY_HI]
NOISY_HI     = 3.02675
NOISY_ZERO   = 5000      # stored value of p_noisy = 0 (exact integer)
NOISY_STEP   = (NOISY_HI - NOISY_LO) / 65535.0       # = 5e-5 log-attenuation per LSB
assert abs(-NOISY_LO / NOISY_STEP - NOISY_ZERO) < 1e-6   # same expression as the generation notebook -> bit-identical encoding

# ---------------------------------------------------------------------------
# Split.
#
# The released membership is the fixed list in splits.csv (seed,split), shipped
# with this repository and read by generate_dataset.py. That file IS the split:
# there is no rule that regenerates it, because membership was fixed before the
# current format and kept unchanged so that every measurement already made stays
# valid.
#
# If splits.csv is missing, generate_dataset.py falls back to a deterministic
# 80/10/10 hash of the seed. That fallback is reproducible but it is NOT the
# released membership, and mixing the two would silently put test patients in
# train. The constant below exists only to name that fallback; it is not a seed
# for the released split and nothing should derive membership from it.
# ---------------------------------------------------------------------------
SPLIT_CSV      = "splits.csv"
FALLBACK_SPLIT_SEED = 42       # fallback hash split only -- NOT the release

# ---------------------------------------------------------------------------
# Patients excluded from evaluation (kept in the dataset for completeness)
# ---------------------------------------------------------------------------
BROKEN_SEEDS = {208084664}   # degenerate VICTRE reconstruction

DENSITIES = ["dense", "hetero", "scattered", "fatty"]

# ---------------------------------------------------------------------------
# Record schema
# ---------------------------------------------------------------------------
N_KEYS = 36          # fields per chunk in v9 (v8 had 34)

# Written into every chunk so a loaded record states which pipeline produced it.
CHANGES_FROM_V8 = ("no penumbra-strip removal; projections and volume normalised "
                   "by the maximum instead of a percentile, so nothing clips; "
                   "half-dose arrays renamed noisy_proj_half and sigma_half; "
                   "strip_valley dropped")

# SSIM slice subsampling. Lives here, not in a script, because run_baselines.py
# (which produced the published table) and evaluate.py (which reusers run) must
# use the SAME value or a reuser's SSIM is not comparable to the paper's.
SSIM_SLICE_STEP = 2      # every 2nd slice -> 28 of 56 scored
