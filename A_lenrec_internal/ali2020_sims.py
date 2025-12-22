import os
import sys
import yaml
import healpy as hp
import numpy as np

sys.path.insert(0, "./")

from library_parameter import *
from utils import bl


# =====================================================
# Load YAML config at import time
# =====================================================
_CONFIG_FILE = os.environ.get("ALI_CONFIG", "config.yaml")

with open(_CONFIG_FILE, "r") as f:
    _cfg = yaml.safe_load(f)


# =====================================================
# Utility
# =====================================================
def map_cut0(nside, map, llow, ltop, lmax=None):
    """
    Cut the map between llow and ltop
    """
    if lmax is None:
        lmax = 3 * nside - 1

    map_cut = (
        hp.alm2map(hp.map2alm(map, lmax=ltop), nside=nside)
        - hp.alm2map(hp.map2alm(map, lmax=llow), nside=nside)
    )
    return map_cut


# =====================================================
# Sims container (parameter-only, no execution)
# =====================================================
class simsLensing:
    """
    Container of simulated / observed maps used for lensing reconstruction.
    All paths are provided by YAML config.
    """

    def __init__(self):
        self.cmbs = _cfg["observation"]["cmbs"]   # list of map path templates

    def hashdict(self):
        """
        Hashable dictionary for reproducibility check
        """
        return {
            "cmbs": self.cmbs
        }

    def replace_fwhm_alm(self, alm, fwhm_old, fwhm_new, pix_old, pix_new, lmax):
        """
        Replace old beam of alm with new beam (arcmin).
        """
        bl_old = bl(fwhm_old, nside=nside, lmax=lmax, pixwin=pix_old)
        bl_new = bl(fwhm_new, nside=nside, lmax=lmax, pixwin=pix_new)
        return hp.almxfl(alm, bl_new / bl_old)

    def replace_fwhm_map(
        self,
        nside,
        lmax,
        map,
        fwhm_old,
        fwhm_new,
        pix_old,
        pix_new,
    ):
        """
        Replace old beam of map with new beam (arcmin).
        """
        alm = hp.map2alm(map)
        alm_new = self.replace_fwhm_alm(
            alm, fwhm_old, fwhm_new, pix_old, pix_new, lmax
        )
        return hp.alm2map(alm_new, nside)

    def get_sim_tmap(self, idx):
        """
        Return band-limited temperature map
        """
        return hp.read_map(self.cmbs[0] % idx, field=0) * hp.read_map(mask_apodiz)

    def get_sim_pmap(self, idx):
        """
        Return band-limited Q/U maps
        """
        Q = hp.read_map(self.cmbs[0] % idx, field=1)
        U = hp.read_map(self.cmbs[0] % idx, field=2)
        mask = hp.read_map(mask_apodiz)
        return Q * mask, U * mask
