import sys
sys.path.insert(0, '../params')
import params as parfile   
from library_parameter import *
import healpy as hp
import numpy as np
import pylab as pl
from plancklens import utils


# 计算 MF, Response, Noise
for qe_key in ['p']:
    print(f"Calculating seeds {seeds}")
    # Calculate mean-field: (This is time-consuming, but only needs to be done once) (与CMB无关，取决于噪声，mask, beam等因素。更换数据后，只需把qlms_dd里的simMF_.fits文件放入即可，并删除全部文件夹内的hash.pk文件)
    qlm_mf = parfile.qlms_dd.get_sim_qlm_mf(qe_key, parfile.mc_sims_mf_dd)
    qresp = parfile.qresp_dd.get_response(qe_key, 'p')
    nhl = parfile.nhl_dd.get_sim_nhl(-1, qe_key, qe_key)

"""
clpp = parfile.cl_unl['pp']
lmax = min([len(clpp) - 1, parfile.qresp_dd.lmax_qlm])
for qe_key in ['p', ]:
    # Calculate mean-field: (This is time-consuming, but only needs to be done once)
    qlm_p_mf = parfile.qlms_dd.get_sim_qlm_mf(qe_key, parfile.mc_sims_mf_dd)
    qresp = parfile.qresp_dd.get_response(qe_key, 'p')
    nhl = parfile.nhl_dd.get_sim_nhl(-1, qe_key, qe_key) * utils.cli(qresp ** 2)
    ll = np.arange(lmax + 1)
    np.savetxt(f"/sharefs/alicpt/users/chenwz/reconstruction_2048_simons2/wiener.dat", [ll, nhl, clpp[:lmax + 1] + nhl])
    print("wiener.dat saved")
"""
"""
# Calculate reconstruction qlm
for qe_key in ['p', ]:
    qlm_p_mf = parfile.qlms_dd.get_sim_qlm_mf(qe_key, parfile.mc_sims_mf_dd)
    qresp = parfile.qresp_dd.get_response(qe_key, 'p')
    for idx in range(0, len(seeds)):
        # Calculate reconstruction for each simulation
        qlm = parfile.qlms_dd.get_sim_qlm(qe_key, idx)
        qlm_rec = hp.almxfl(qlm-qlm_p_mf, utils.cli(qresp))
        hp.write_alm(f"/sharefs/alicpt/users/chenwz/reconstruction_2048_simons2/rec_qlm/rec_alm_{idx}.fits", qlm_rec, overwrite=True)
        print(f"rec_alm_{idx}.fits saved")
"""