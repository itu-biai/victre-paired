"""
VICTRE-Paired — noise model (data format v9; unchanged from v8).

Noisy projections are generated from the *released, quantised* `clean_proj`, so
every `noisy_proj*` array is reproducible bit-for-bit from the record itself:

    p      = clean_proj_u16 / 65535                      (log-attenuation, [0,1])
                                                         NOTE: clean_proj uses /65535,
                                                         noisy_proj* use (u16-5000)*5e-5
    I0     = 1 / gain                                    photons per pixel per view
    I      = I0 * exp(-p * proj_scale)                   expected transmitted photons
    N      = Poisson(I) + Normal(0, ELEC_PHOTONS)        detected photons
    p_n    = -log(max(N, PHOTON_FLOOR) / I0) / proj_scale
    p_n    = 0 outside the illuminated bounding box of clean_proj > 0 (per view)
    stored = uint16( round(p_n / NOISY_STEP) + NOISY_ZERO )   -> p_n = (u16 - NOISY_ZERO) * NOISY_STEP

RNG: numpy default_rng(seed*10 + dose_idx), dose_idx full:0 half:1 quarter:2.
Nothing is clipped to [0,1]: negative values in air are kept (no pedestal) and the
thickest tissue cannot saturate (max representable 3.02675 > max possible ~2.9 in v9,
where the high-attenuation band removed by v8's threshold is back).

`regenerate(cp_u16, proj_scale, seed, dose)` is the only function a user needs: it
returns the stored uint16 array for one patient and one budget, bit for bit.
"""
import numpy as np
from constants import (DOSES, DOSE_IDX, ELEC_PHOTONS, PHOTON_FLOOR,
                       NOISY_LO, NOISY_HI, NOISY_ZERO, NOISY_STEP)


def bbox_mask(cp_u16):
    """Per-view bounding box of the illuminated detector region, from clean_proj > 0."""
    m = np.zeros(cp_u16.shape, bool)
    for v in range(cp_u16.shape[0]):
        ys, xs = np.where(cp_u16[v] > 0)
        if len(ys):
            m[v, ys.min():ys.max()+1, xs.min():xs.max()+1] = True
    return m


def add_noise(cp_u16, proj_scale, gain, seed, dose_idx):
    """Noisy log-attenuation (float32, unclipped except to [NOISY_LO, NOISY_HI])."""
    p = cp_u16.astype(np.float32) / 65535.0
    rng = np.random.default_rng(np.uint64(seed) * 10 + np.uint64(dose_idx))
    I0 = 1.0 / gain
    I = I0 * np.exp(-p * proj_scale)
    N = rng.poisson(I).astype(np.float64) + rng.standard_normal(p.shape) * ELEC_PHOTONS
    pn = (-np.log(np.maximum(N, PHOTON_FLOOR) / I0) / proj_scale).astype(np.float32)
    pn = np.where(bbox_mask(cp_u16), pn, 0.0).astype(np.float32)
    return np.clip(pn, NOISY_LO, NOISY_HI)


def encode(pn):
    """float log-attenuation -> stored uint16 (exact integer zero level)."""
    return np.clip(np.round(pn / NOISY_STEP) + NOISY_ZERO, 0, 65535).astype(np.uint16)


def decode(u16):
    """stored uint16 -> float32 log-attenuation."""
    return ((np.asarray(u16).astype(np.float32) - NOISY_ZERO) * np.float32(NOISY_STEP)).astype(np.float32)


def regenerate(cp_u16, proj_scale, seed, dose):
    """Stored uint16 noisy array for one patient and one photon budget ('full'|'half'|'quarter')."""
    return encode(add_noise(cp_u16, proj_scale, DOSES[dose], seed, DOSE_IDX[dose]))
