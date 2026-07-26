"""
geometry.py — LEAP forward / adjoint projection operators for the VICTRE-PAIRED
v6 acquisition geometry (modular-beam, +/-25 degrees, 25 views).

Requires LEAP (github.com/LLNL/LEAP) and a CUDA GPU:
    git clone --depth 1 https://github.com/LLNL/LEAP.git && pip install ./LEAP
"""
import gc

import numpy as np
import torch
from leapctype import tomographicModels

from constants import (
    SID, SDD, DET_PIX, NA, ANGLE_DEG, PROJ_ROWS, PROJ_COLS,
    RECON_Z, RECON_H, RECON_W, VOX_XY, VOX_Z,
)


class Projector:
    """LEAP modular-beam model with normalized forward (A) and adjoint (AT)
    operators.

    The forward operator is scaled by PROJ_SCALE so that a volume of ones
    projects to a maximum value of 1, matching the min-max normalized
    reconstructions stored in the dataset. `L` is the underlying
    tomographicModels instance, exposed so iterative solvers (SIRT, SART,
    ASD-POCS) can be called directly on unnormalized projections `g * PROJ_SCALE`.
    """

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ZN, self.NY, self.NX = RECON_Z, RECON_W, RECON_H
        I2D = SDD - SID
        th = np.radians(np.linspace(-ANGLE_DEG, ANGLE_DEG, NA))

        L = tomographicModels()
        L.set_modularbeam(
            NA, PROJ_ROWS, PROJ_COLS, DET_PIX, DET_PIX,
            np.stack([SID * np.sin(th), np.zeros(NA), SID * np.cos(th)], 1).astype(np.float32),
            np.tile(np.array([0, 0, -I2D], np.float32), (NA, 1)),
            np.tile(np.array([1, 0, 0], np.float32), (NA, 1)),
            np.tile(np.array([0, 1, 0], np.float32), (NA, 1)),
        )
        L.set_volume(RECON_H, RECON_W, RECON_Z, VOX_XY, VOX_Z,
                     0.0, 0.0, -I2D + (RECON_Z * VOX_Z) / 2.0)
        try:
            L.set_log_error()
        except Exception:
            pass
        self.L = L

        # calibrate PROJ_SCALE from a volume of ones
        with torch.no_grad():
            f = torch.ones(RECON_Z, RECON_H, RECON_W, device=self.device)
            g = torch.zeros((NA, PROJ_ROWS, PROJ_COLS), device=self.device)
            L.project(g, f.permute(0, 2, 1).contiguous())
        self.PROJ_SCALE = float(g.max().item())
        del f, g
        gc.collect(); torch.cuda.empty_cache()

    def A(self, f):
        """Normalized forward projection: volume (ZN,H,W) -> projections (NA,R,C)."""
        g = torch.zeros((NA, PROJ_ROWS, PROJ_COLS), device=f.device)
        self.L.project(g, f.detach().permute(0, 2, 1).contiguous())
        return g / self.PROJ_SCALE

    def AT(self, g):
        """Normalized adjoint (back-projection): projections -> volume (ZN,H,W)."""
        gf = torch.zeros((self.ZN, self.NY, self.NX), device=g.device)
        self.L.backproject((g.detach() / self.PROJ_SCALE).contiguous(), gf)
        return gf.permute(0, 2, 1).contiguous()

    def adjoint_ratio(self):
        """<A f, g> / <f, AT g> on random inputs; should be ~1 if A, AT are adjoint."""
        with torch.no_grad():
            f = torch.rand(self.ZN, RECON_H, RECON_W, device=self.device)
            g = torch.rand(NA, PROJ_ROWS, PROJ_COLS, device=self.device)
            return float(((self.A(f) * g).sum() / (f * self.AT(g)).sum()).item())
