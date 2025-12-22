import os
import yaml

# -------------------------------------------------
# Load config
# -------------------------------------------------
_CONFIG_FILE = os.environ.get("ALI_CONFIG", "config.yaml")
with open(_CONFIG_FILE, "r") as f:
    _cfg = yaml.safe_load(f)


# -------------------------------------------------
# QE settings
# -------------------------------------------------
# lib_qe_keys = {
#     k: [v["color"], v["label"]]
#     for k, v in _cfg["qe"]["lib_qe_keys"].items()
# }


# -------------------------------------------------
# Paths
# -------------------------------------------------
ALILENS = _cfg["paths"]["ALILENS"]


# -------------------------------------------------
# Map / harmonic parameters
# -------------------------------------------------
nside = _cfg["map"]["nside"]
lmax = 3 * nside - 1
dlmax = _cfg["map"]["dlmax"]


# -------------------------------------------------
# Simulation control
# -------------------------------------------------
seeds = list(range(_cfg["simulation"]["seeds"]))
bias = _cfg["simulation"]["bias"]
var = _cfg["simulation"]["var"]
nset = _cfg["simulation"]["nset"]
nwidth = _cfg["simulation"]["nwidth"]


# -------------------------------------------------
# Noise
# -------------------------------------------------
nlev = _cfg["noise"]["nlev"]
fwhm_f = _cfg["noise"]["fwhm_f"]


# -------------------------------------------------
# Masks
# -------------------------------------------------
mask_bb = _cfg["mask"]["bb"]
mask_apodiz = _cfg["mask"]["apodiz"]

