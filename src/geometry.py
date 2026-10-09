"""
VICTRE-Paired — projection geometry and reconstruction operators.

Wraps a LEAP modular-beam geometry built from the per-patient geom_* fields, and
exposes forward / adjoint projection plus the classical reconstruction methods
used by run_baselines.py. A CUDA GPU and the `leapctype` package are required.

Example
-------
    import numpy as np
    from geometry import Geometry

    d = np.load("test/test_chunk_00000.npz")
    i = 0
    G = Geometry(vox_z=float(d["geom_vox_z"][i]),
                 offx=float(d["geom_offx"][i]),
                 offy=float(d["geom_offy"][i]),
                 offz=float(d["geom_offz"][i]))
    proj  = d["clean_proj"][i].astype(np.float32) / 65535
    recon = G.fbp(proj)                     # (56, 408, 336), RAW scale

Every reconstruction method returns the raw volume; compare it to the reference
with a scale-invariant metric (masked correlation) or after an affine fit inside
the breast mask (see run_baselines.py).
"""

import math
import numpy as np
import torch
from leapctype import tomographicModels

from constants import (NA, PH, PW, DET_PIX, ANG, SID, SDD,
                       ZOUT, TH, TW, VOX_XY)

DEV = "cuda"


class Geometry:
    """LEAP modular-beam geometry + forward/adjoint + FBP/ATp/SIRT/SART/ASD-POCS.

    The trajectory (25 views over ±25°), detector and SID/SDD are fixed for the
    whole dataset; only the volume placement (vox_z, offx, offy, offz) varies
    per patient and comes from the stored geom_* fields.
    """

    def __init__(self, vox_z, offx, offy, offz):
        th = np.radians(np.linspace(-ANG, ANG, NA))
        L = tomographicModels()
        L.set_modularbeam(
            NA, PH, PW, DET_PIX, DET_PIX,
            np.stack([SID * np.sin(th), np.zeros(NA), SID * np.cos(th)], 1).astype(np.float32),
            np.tile(np.array([0, 0, -(SDD - SID)], np.float32), (NA, 1)),
            np.tile(np.array([1, 0, 0], np.float32), (NA, 1)),
            np.tile(np.array([0, 1, 0], np.float32), (NA, 1)))
        L.set_volume(TH, TW, ZOUT, VOX_XY, float(vox_z),
                     float(offx), float(offy), float(offz))
        try:
            L.set_log_error()          # silence LEAP's per-iteration prints
        except AttributeError:
            # Only an old/renamed LEAP API is tolerated here. A bare `except
            # Exception: pass` would also swallow a CUDA or geometry failure and
            # leave a silently broken object behind.
            print("[geometry] note: LEAP has no set_log_error(); "
                  "per-iteration output stays on")
        self.L = L
        # scale factor so that A(ones) has unit max (keeps metrics comparable)
        with torch.no_grad():
            f = torch.ones(ZOUT, TH, TW, device=DEV)
            g = torch.zeros((NA, PH, PW), device=DEV)
            L.project(g, f.permute(0, 2, 1).contiguous())
        # Scale convention. `ps = max(A(ones))` converts between two units:
        #   LEAP units          - what project()/backproject() work in
        #   unit-max units      - what A() returns and what the released
        #                         `clean_proj` is comparable to
        # A() divides by ps, so every reconstruction method multiplies its input
        # by ps to invert that. Before this was made uniform, SIRT/SART/ASD-POCS
        # scaled the input but ATp and FBP did not, so the same projection came
        # back at two different scales depending on the method.
        self.ps = float(g.max().item())
        del f, g
        torch.cuda.empty_cache()

    # -- core operators -----------------------------------------------------
    def _A(self, vol_t):
        g = torch.zeros((NA, PH, PW), device=DEV)
        with torch.no_grad():
            self.L.project(g, vol_t.permute(0, 2, 1).contiguous())
        return g

    def _AT(self, g):
        fb = torch.zeros(ZOUT, TW, TH, device=DEV)
        with torch.no_grad():
            self.L.backproject(g, fb)
        return fb.permute(0, 2, 1)

    def A(self, vol_np):
        """Forward project a volume → (25, 752, 384), unit-max scaled."""
        t = torch.from_numpy(np.ascontiguousarray(vol_np)).float().to(DEV)
        g = self._A(t) / self.ps
        out = g.cpu().numpy()
        del t, g
        torch.cuda.empty_cache()
        return out

    # -- reconstruction methods --------------------------------------------
    #
    # Every method takes a projection in unit-max units (what A() returns, and
    # what `clean_proj / 65535` is) and returns the RAW reconstruction. Nothing
    # is normalised here.
    #
    # These used to end with a min-max rescale to [0, 1]. That made every
    # radiometric metric (PSNR, SSIM, RMSE) a function of the two extreme voxels
    # rather than of the reconstruction, and it silently differed between methods
    # because the input scaling was not uniform. Normalisation now happens once,
    # in the metric layer, where it is a documented affine fit inside the breast
    # mask (see run_baselines.py: affine_match).

    def atp(self, proj_np):
        """Adjoint (back-projection), Aᵀp. Returns the raw volume."""
        g = torch.from_numpy(np.ascontiguousarray(proj_np * self.ps)).float().to(DEV)
        x = self._AT(g)
        out = x.cpu().numpy()
        del g, x
        torch.cuda.empty_cache()
        return out

    def sirt(self, proj_np, n=50):
        g = torch.from_numpy(np.ascontiguousarray(proj_np * self.ps)).float().to(DEV)
        f = torch.zeros(ZOUT, TW, TH, device=DEV)
        self.L.SIRT(g.contiguous(), f, n)
        out = f.permute(0, 2, 1).contiguous().cpu().numpy()
        del g, f
        torch.cuda.empty_cache()
        return out

    def sart(self, proj_np, n=4):
        g = torch.from_numpy(np.ascontiguousarray(proj_np * self.ps)).float().to(DEV)
        f = torch.zeros(ZOUT, TW, TH, device=DEV)
        self.L.SART(g.contiguous(), f, n)
        out = f.permute(0, 2, 1).contiguous().cpu().numpy()
        del g, f
        torch.cuda.empty_cache()
        return out

    def asdpocs(self, proj_np, n_asd=20, n_sub=1, n_tv=20):
        g = torch.from_numpy(np.ascontiguousarray(proj_np * self.ps)).float().to(DEV)
        f = torch.zeros(ZOUT, TW, TH, device=DEV)
        self.L.ASDPOCS(g.contiguous(), f, n_asd, n_sub, n_tv)
        out = f.permute(0, 2, 1).contiguous().cpu().numpy()
        del g, f
        torch.cuda.empty_cache()
        return out

    def fbp(self, proj_np, window="hann", cosine_weight=True):
        """Filtered back-projection with an explicit ramp filter.

        LEAP's built-in FBP does not work on this modular-beam DBT geometry, so
        the ramp filter is applied by hand (Hann-windowed, edge-replicate padded)
        and back-projected via the adjoint. This is independent of VICTRE's own
        reconstruction code.
        """
        g = np.asarray(proj_np, np.float32).copy()
        if cosine_weight:
            u = (np.arange(PH) - PH / 2.0) * DET_PIX
            v = (np.arange(PW) - PW / 2.0) * DET_PIX
            w = SID / np.sqrt(SID**2 + u[:, None]**2 + v[None, :]**2)
            g *= w[None].astype(np.float32)
        n = int(2 ** math.ceil(math.log2(PH * 1.6)))
        pl = (n - PH) // 2
        pr = n - PH - pl
        gp = np.pad(g, ((0, 0), (pl, pr), (0, 0)), mode="edge")
        fr = np.fft.rfftfreq(n)
        H = 2.0 * fr
        if window == "hann":
            H = H * (0.5 + 0.5 * np.cos(np.pi * fr / max(fr.max(), 1e-9)))
        gf = np.fft.irfft(np.fft.rfft(gp, axis=1) * H[None, :, None],
                          n=n, axis=1)[:, pl:pl + PH, :].astype(np.float32)
        t = torch.from_numpy(np.ascontiguousarray(gf * self.ps)).float().to(DEV)
        x = self._AT(t)
        out = x.cpu().numpy()
        del t, x
        torch.cuda.empty_cache()
        return out


def geometry_from_chunk(d, i):
    """Convenience: build a Geometry for patient `i` in a loaded chunk `d`."""
    return Geometry(vox_z=float(d["geom_vox_z"][i]),
                    offx=float(d["geom_offx"][i]),
                    offy=float(d["geom_offy"][i]),
                    offz=float(d["geom_offz"][i]))
