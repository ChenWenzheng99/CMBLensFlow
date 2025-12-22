import os
import sys
import yaml
import threading

from lenspyx.utils import camb_clfile
import lenspyx

sys.path.insert(0, "./")
from utils import bl_eft
from library_parameter import *

# =========================
# load YAML config at import time
# =========================
_CONFIG_FILE = os.environ.get("ALI_CONFIG", "config.yaml")

with open(_CONFIG_FILE, "r") as f:
    _cfg = yaml.safe_load(f)




transf = bl_eft(nlev, fwhm_f, lmax=lmax, pixwin=True)


# =========================
# input Cls
# =========================
cls_path = os.path.join(os.path.dirname(os.path.abspath(lenspyx.__file__)), 'data', 'cls')
cls_in = camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))


# =========================
# reconstruction parameters
# =========================
_rcfg = _cfg["reconstruction"]

if _rcfg.get("nsims_from_seeds", True):
    nsims = len(seeds) - 1
else:
    nsims = _rcfg["nsims"]

lmin_ivf = _rcfg["lmin_ivf"]
lmax_ivf = _rcfg["lmax_ivf"]
lmax_qlm = _rcfg["lmax_qlm"]


# =========================
# paths
# =========================
_base = _cfg["paths"]["ALILENS"]

savePath_cmbs  = os.path.join(_base, "sims", "cmbs")
savePath_noise = os.path.join(_base, "sims", "noise")
savePath_ninv  = os.path.join(_base, "sims", "ninv")

libdir_cinvt   = os.path.join(_base, "temp", "cinv_t")
libdir_cinvp   = os.path.join(_base, "temp", "cinv_p")
libdir_ivfs    = os.path.join(_base, "temp", "ivfs")
libdir_qlms_dd = os.path.join(_base, "temp", "qlms_dd")
libdir_qlms_ds = os.path.join(_base, "temp", "qlms_ds")
libdir_qlms_ss = os.path.join(_base, "temp", "qlms_ss")
libdir_qcls_dd = os.path.join(_base, "temp", "qcls_dd")
libdir_qcls_ds = os.path.join(_base, "temp", "qcls_ds")
libdir_qcls_ss = os.path.join(_base, "temp", "qcls_ss")
libdir_nhl_dd  = os.path.join(_base, "temp", "nhl_dd")
libdir_n1_dd   = os.path.join(_base, "temp", "n1_dd")
libdir_qresp   = os.path.join(_base, "temp", "qresp")


# =========================
# thread-safe mkdir
# =========================
_lock = threading.Lock()

def _mkdir(path):
    if not os.path.exists(path):
        with _lock:
            if not os.path.exists(path):
                os.makedirs(path)

for _p in [
    savePath_cmbs, savePath_noise, savePath_ninv,
    libdir_cinvt, libdir_cinvp, libdir_ivfs,
    libdir_qlms_dd, libdir_qlms_ds, libdir_qlms_ss,
    libdir_qcls_dd, libdir_qcls_ds, libdir_qcls_ss,
    libdir_nhl_dd, libdir_n1_dd, libdir_qresp,
]:
    _mkdir(_p)
