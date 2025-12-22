import sys
sys.path.insert(0, '../params')
import params as parfile   
from library_parameter import *
import healpy as hp
import numpy as np
import pylab as pl
from plancklens import utils



_CONFIG_FILE = os.environ.get("ALI_CONFIG", "config.yaml")
with open(_CONFIG_FILE, "r") as f:
    _cfg = yaml.safe_load(f)


clpp = parfile.cl_unl['pp']
lmax = min([len(clpp) - 1, parfile.qresp_dd.lmax_qlm])
for qe_key in [_cfg['qe']["lib_qe_keys"][0]]:   #'ptt', 
    print(qe_key)
    # Calculate mean-field: (This is time-consuming, but only needs to be done once)
    #qlm_p_mf = parfile.qlms_dd.get_sim_qlm_mf(qe_key, parfile.mc_sims_mf_dd)
    qresp = parfile.qresp_dd.get_response(qe_key, 'p')
    nhl = parfile.nhl_dd.get_sim_nhl(-1, qe_key, qe_key) * utils.cli(qresp ** 2)
    ll = np.arange(lmax + 1)
    np.savetxt(f"wiener_red.dat", [ll, nhl, clpp[:lmax + 1] + nhl])
    print("wiener.dat saved.")




for qe_key in [_cfg['qe']["lib_qe_keys"][0]]:
    qlm_p_mf = parfile.qlms_dd.get_sim_qlm_mf(qe_key, parfile.mc_sims_mf_dd)
    qresp = parfile.qresp_dd.get_response(qe_key, 'p')
    for idx in range(0, len(seeds)):
        # Calculate reconstruction for each simulation
        qlm = parfile.qlms_dd.get_sim_qlm(qe_key, idx)
        qlm_rec = hp.almxfl(qlm-qlm_p_mf, utils.cli(qresp))
        hp.write_alm(f"{_cfg['rec_path']['qlm_rec']}/qlm_QE_{qe_key}_{idx:04d}.fits", qlm_rec, overwrite=True)
        print(f"rec_alm_{idx}.fits saved.")

