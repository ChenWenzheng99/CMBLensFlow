"""To enable complete reconstruction, a parameter file should instantiate

        * the inverse-variance filtered simulation library 'ivfs'
        * the 3 quadratic estimator libraries, 'qlms_dd', 'qlms_ds', 'qlms_ss'.
        * the 3 quadratic estimator power spectra libraries 'qcls_dd, 'qcls_ds', 'qcls_ss'.
          (qcls_ss is required for the MCN0 calculation, qcls_ds and qcls_ss for the RDN0 calculation.
           qcls_dd for the MC-correction, covariance matrix. All three for the point source correction.)
        * the quadratic estimator response library 'qresp_dd'
        * the semi-analytical Gaussian lensing bias N0 library 'nhl_dd'
        * the N1 lensing bias library 'n1_dd'.

    The module bandpowers.py shows how these elements are used to build the reconstructed bandpowers.

    On the first call this module will cache a couple of things will be cached in the directories defined below.

    NOTE Conjugate gradient inversion method is used here instead of Homogeneous filtering or that with rescaling.
         Apodization mask is not a must in Conjugate gradient inversion.
         Plus you dont have to specify the white noise level too.

         
#############################################################该文件参考plancklens 的idealized_example.py#########################################################
仅处理均匀的,各向同性的white noise
"""

import os
import healpy as hp
import numpy as np
import sys

import plancklens
from plancklens.filt import filt_util, filt_cinv
from plancklens import utils
from plancklens import qest, qecl, qresp
from plancklens import nhl
from plancklens.n1 import n1
from plancklens.sims import utils as maps_utils
from plancklens.qcinv import cd_solve


sys.path.insert(0, './')
from ali2020_sims import simsLensing
from utils import bl_eft
from one import *
from library_parameter import *


# Data Paths  (cl是data, ninv是由I_NOISE_150_C_1024.fits 的 noiselevel 得到 Noise inveres pixel varance)
cls_path = os.path.join(os.path.dirname(os.path.abspath(plancklens.__file__)), 'data', 'cls')  #/sharefs/alicpt/users/chenwz/download/plancklens/plancklens/data/cls
ninv_t_Path = os.path.join(ALILENS, 'sims/ninv/ninv_t.fits')
ninv_p_Path = os.path.join(ALILENS, 'sims/ninv/ninv_p.fits')



# Transfer function and Input power spectrum
cl_unl = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
cl_len = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
#cl_len = utils.camb_clfile('/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MAIN/revise/rec/QE_HO_temperature_SO/mmDL_lensedCls_sm.dat')



# CMB spectra entering the QE weights (the spectra multplying the inverse-variance filtered maps in the QE legs)
cl_weight = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
#cl_weight = utils.camb_clfile('/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MAIN/revise/rec/QE_HO_temperature_SO/mmDL_lensedCls_sm.dat')
cl_weight['bb'] *= 0.


# Simulation library for Ali noise
# NOTE in planck there is extra power dcl to better match the data properties
sims = simsLensing()
sims = maps_utils.sim_lib_shuffle(sims, { idx : nsims if idx == -1 else idx for idx in range(-1, nsims) })  #idx=-1为data,放在最后199；idx>0为sims,放在0到nsims-1(0到198)


# Preconditioner
"""
for [id, pre_ops_descr, lmax, nside, iter_max, eps_min, tr, cache] in chain_descr
by default the chain_descr =

    [[3, ["split(dense(" + pcf + "), 64, diag _cl)"],    256,    128,      3,    0.0, cd_solve.tr_cg, cd_solve.cache_mem()],
     [2, ["split(stage(3),  256, diag_cl)"],            512,    256,      3,    0.0, cd_solve.tr_cg, cd_solve.cache_mem()],
     [1, ["split(stage(2),  512, diag_cl)"],            1024,   512,      3,    0.0, cd_solve.tr_cg, cd_solve.cache_mem()],
     [0, ["split(stage(1), 1024, diag_cl)"],            lmax, nside, np.inf, 1.0e-5, cd_solve.tr_cg, cd_solve.cache_mem()]]

"""
# Julien's suggested eps_min = 1e-3 or 1e-4
chain_descr = [[0, ["diag_cl"], lmax_ivf, nside, np.inf, 1e-4, cd_solve.tr_cg, cd_solve.cache_mem()]]
chain_descr2 = [[0, ["diag_cl"], lmax_ivf, nside, np.inf, 1e-3, cd_solve.tr_cg, cd_solve.cache_mem()]]

# Conjuage inversion
ninv_t = [ninv_t_Path]    #Inverse variance map,由sims.py的ninv函数生成
#Temperature-only inverse-variance (or Wiener-)filtering instance
cinv_t = filt_cinv.cinv_t(libdir_cinvt, lmax_ivf, nside, cl_len, transf, ninv_t, marge_monopole=True, marge_dipole=True, marge_maps=[], chain_descr=chain_descr)

ninv_p = [[ninv_p_Path]]  #Inverse variance map,由sims.py的ninv函数生成
#Polarization-only inverse-variance (or Wiener-)filtering instance
cinv_p = filt_cinv.cinv_p(libdir_cinvp, lmax_ivf, nside, cl_len, transf, ninv_p, chain_descr=chain_descr2) 

#Library to perform inverse-variance filtering of a simulation library.
ivfs_raw = filt_cinv.library_cinv_sepTP(libdir_ivfs, sims, cinv_t, cinv_p, cl_len) 

# rescaling or cuts. Here just a lmin cut
ftl_rs = (np.arange(lmax_ivf + 1) >= lmin_ivf)  #filtered temperature alms are rescaled by ftl_rs,一个bool值列表
fel_rs = (np.arange(lmax_ivf + 1) >= lmin_ivf)  #filtered E-polarization alms are rescaled by fel_rs,一个bool值列表
fbl_rs = (np.arange(lmax_ivf + 1) >= lmin_ivf)  #filtered B-polarization alms are rescaled by fbl_rs,一个bool值列表
ivfs   = filt_util.library_ftl(ivfs_raw, lmax_ivf, ftl_rs, fel_rs, fbl_rs) #Library of a-posteriori re-scaled filtered CMB maps
#只是借用了截断l的功能，因为观测天区范围有限，在ivfs时也要对应地去算，可见PLANCK2018 P21.

# QE libraries instances.
# For the MCN0, RDN0, MC-correction etc calculation, we need in general three of them,
# qlms_dd is the QE library which builds a lensing estimate with the same simulation on both legs，仅一套作为data, 用不到shuffle, 是同一个simulation.
# qlms_ds is the QE library which builds a lensing estimate with a simulation on one leg and the data on the second.
# qlms_ss is the QE library which builds a lensing estimate with a simulation on one leg and another simulation on the second.


# Shuffling dictionary.
# ss_dict remaps idx -> idx + 1 by blocks of 60 up to 300.
ss_dict = { k : v for k, v in zip( np.arange(nsims), np.concatenate([np.roll(range(i*nwidth, (i+1)*nwidth), -1) for i in range(0,nset)]))} 
ds_dict = { k : -1 for k in range(nsims) }   #k代表各个模拟, -1代表数据
ivfs_d = filt_util.library_shuffle(ivfs, ds_dict) # This is a filtering instance always returning the data map.
ivfs_s = filt_util.library_shuffle(ivfs, ss_dict) # This is a filtering instance shuffling simulation indices according to 'ss_dict'.

#QE estimators instances (from data-data, data-sim, sim-sim)
qlms_dd = qest.library_sepTP(libdir_qlms_dd, ivfs, ivfs,   cl_len['te'], nside, lmax_qlm=lmax_qlm) #见/home/rabbit/settings/plancklens/plancklens/qest.py
qlms_ds = qest.library_sepTP(libdir_qlms_ds, ivfs, ivfs_d, cl_len['te'], nside, lmax_qlm=lmax_qlm)
qlms_ss = qest.library_sepTP(libdir_qlms_ss, ivfs, ivfs_s, cl_len['te'], nside, lmax_qlm=lmax_qlm)
#Ffp10才是真正计算的地方，params.py只是建立library instances.
#Cl_len[‘te’]见PLANCK2018: 即计算T-only和P-only是忽略Cl^TE,而只用到其他Cl. 但是算MV时还需要考虑，故给出该参数


#ss_dict共9组，每组22对(前21对让k和v错位1，第22对把k、v分别剩余的一个配对)(保证了两个leg的simulation不同，见PLANCK2018(A.1)下面的话，''cyclically'')
#0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 0,
#22: 23, 23: 24, 24: 25, 25: 26, 26: 27, 27: 28, 28: 29, 29: 30, 30: 31, 31: 32, 32: 33, 33: 34, 34: 35, 35: 36, 36: 37, 37: 38, 38: 39, 39: 40, 40: 41, 41: 42, 42: 43, 43: 22,
#44: 45, 45: 46, 46: 47, 47: 48, 48: 49, 49: 50, 50: 51, 51: 52, 52: 53, 53: 54, 54: 55, 55: 56, 56: 57, 57: 58, 58: 59, 59: 60, 60: 61, 61: 62, 62: 63, 63: 64, 64: 65, 65: 44,
#66: 67, 67: 68, 68: 69, 69: 70, 70: 71, 71: 72, 72: 73, 73: 74, 74: 75, 75: 76, 76: 77, 77: 78, 78: 79, 79: 80, 80: 81, 81: 82, 82: 83, 83: 84, 84: 85, 85: 86, 86: 87, 87: 66,
#88: 89, 89: 90, 90: 91, 91: 92, 92: 93, 93: 94, 94: 95, 95: 96, 96: 97, 97: 98, 98: 99, 99: 100, 100: 101, 101: 102, 102: 103, 103: 104, 104: 105, 105: 106, 106: 107, 107: 108, 108: 109, 109: 88,
#110: 111, 111: 112, 112: 113, 113: 114, 114: 115, 115: 116, 116: 117, 117: 118, 118: 119, 119: 120, 120: 121, 121: 122, 122: 123, 123: 124, 124: 125, 125: 126, 126: 127, 127: 128, 128: 129, 129: 130, 130: 131, 131: 110,
#132: 133, 133: 134, 134: 135, 135: 136, 136: 137, 137: 138, 138: 139, 139: 140, 140: 141, 141: 142, 142: 143, 143: 144, 144: 145, 145: 146, 146: 147, 147: 148, 148: 149, 149: 150, 150: 151, 151: 152, 152: 153, 153: 132,
#154: 155, 155: 156, 156: 157, 157: 158, 158: 159, 159: 160, 160: 161, 161: 162, 162: 163, 163: 164, 164: 165, 165: 166, 166: 167, 167: 168, 168: 169, 169: 170, 170: 171, 171: 172, 172: 173, 173: 174, 174: 175, 175: 154,
#176: 177, 177: 178, 178: 179, 179: 180, 180: 181, 181: 182, 182: 183, 183: 184, 184: 185, 185: 186, 186: 187, 187: 188, 188: 189, 189: 190, 190: 191, 191: 192, 192: 193, 193: 194, 194: 195, 195: 196, 196: 197, 197: 176}

#---- QE spectra libraries instances:
# This takes power spectra of the QE maps from the QE libraries, after subtracting a mean-field.
# Only qcls_dd needs a mean-field subtraction.(For qcls_ds and qcls_ss, the mean field would vanish, since the different maps are independent to a very good approximation)

# qecl libraries instances
mc_sims_bias = np.arange(bias) #: The mean-field will be calculated from these simulations.
mc_sims_var  = np.arange(bias, var+bias) #: The covariance matrix will be calculated from these simulations

# Only qcls_dd needs a mean-field subtraction.
mc_sims_mf_dd = mc_sims_bias
mc_sims_mf_ds = np.array([])
mc_sims_mf_ss = np.array([])

#QE estimators (cross-)power spectra library template (对于cl^dd,生成cl after mean-filed subtraction)
qcls_dd = qecl.library(libdir_qcls_dd, qlms_dd, qlms_dd, mc_sims_mf_dd) #即Cl^dd,亦即Cl^ii(-1,-1)
qcls_ds = qecl.library(libdir_qcls_ds, qlms_ds, qlms_ds, mc_sims_mf_ds) #即Cl^ds,亦即Cl_di(i,-1)
qcls_ss = qecl.library(libdir_qcls_ss, qlms_ss, qlms_ss, mc_sims_mf_ss) #即Cl^ss,亦即Cl^ij(i,j; j=i+1,cyclically)


# Semi-analytical Gaussian lensing bias library:即N0
nhl_dd = nhl.nhl_lib_simple(libdir_nhl_dd, ivfs, cl_weight, lmax_qlm) #Semi-analytical unnormalized N0 library

# N1 lensing bias library:
n1_dd = n1.library_n1(libdir_n1_dd, cl_len['tt'],cl_len['te'],cl_len['ee']) #Flexible library for calculation of the N1 quadratic estimator biases

# QE response calculation library:
qresp_dd = qresp.resp_lib_simple(libdir_qresp, lmax_ivf, cl_weight, cl_len, {'t': ivfs.get_ftl(), 'e':ivfs.get_fel(), 'b':ivfs.get_fbl()}, lmax_qlm) #QE responses calculation library
