from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

import sys
sys.path.insert(0, '../params')
import params as parfile   
from library_parameter import *
import healpy as hp
import numpy as np
import pylab as pl
from plancklens import utils
import generate_LSS as glss

sys.path.append('/sharefs/alicpt/users/chenwz/download/cmblensplus2/utils')
sys.path.append('/sharefs/alicpt/users/chenwz/Testarea/FORECAST/Foreground/foreground_run')

from utils_mine import *

##########################################################################
##########################    Load map dir     ###########################

# 获取传递给脚本的参数
args = sys.argv[1:]

# 打印当前作业的编号
# 计算当前作业的文件编号列表
#file_indices = comm_rank + int(args[0])
file_indices = 442  # 112号LSS sim有问题, 单独跑

# 打印当前作业的文件编号列表
print(f"Job {comm_rank} will process files: {file_indices}")

# map读取路径 /map_TQU_2048_0000.fits
rec_qlm_filename = f"/sharefs/alicpt/users/chenwz/reconstruction_multi_process2/process_added/com_rec/rec_qlm_{file_indices}.fits"
P_filename = f"/sharefs/alicpt/users/chenwz/reconstruction_2048_simons/test/ALILENS/sims/cmbs/map_P_2048_{file_indices:04d}.fits"

nlpp = np.loadtxt('/sharefs/alicpt/users/chenwz/reconstruction_2048_simons/wiener.dat')[1]

tracers_filename = f"/sharefs/alicpt/users/chenwz/reconstruction_multi_process2/process_added/LSS_tracer/Kappa_CIB_LSST/tracers_KIL_{file_indices}.npy"
comb_filename = f"/sharefs/alicpt/users/chenwz/reconstruction_multi_process2/process_added/LSS_tracer/COMB_KIL/comb_KIL_{file_indices}.npy"

rho_filename = f"/sharefs/alicpt/users/chenwz/reconstruction_multi_process2/process_added/LSS_tracer/rhos/rhos_{file_indices}.npz"
rho_fig_filename = f"/sharefs/alicpt/users/chenwz/reconstruction_multi_process2/process_added/LSS_tracer/rhos_fig/rhos_fig_{file_indices}.png"

nside = 2048
lmax = 3*1024-1
cls = np.load('/sharefs/alicpt/users/chenwz/reconstruction_multi_process2/process_added/k_I_LSST_power_matrix_with_sn_new.npy')

mask = hp.read_map('/sharefs/alicpt/users/chenwz/Testarea/FORECAST/mask/AliCPT_20uKcut150_C_2048.fits')
fsky = calc_fsky(nside,mask,mask)

surveys = ['kappa','CIB','LSST1','LSST2','LSST3','LSST4','LSST5','LSST6','LSST7',]

fsky = calc_fsky(nside,mask,mask)

q2k = lambda l: l*(l + 1) / 2 # potential -> convergence
q2d = lambda l: (l*(l + 1)) ** 0.5 # potential -> deflection
cut = np.where((np.arange(lmax + 1) > 10) * (np.arange(lmax + 1) < 3072), 1, 0) # band limit

def view_map(m, title, min=None, max=None, cmap='YlGnBu_r'):
     """ View map.
     """
     # TODO beautify this plot
     rot = [180, 60, 0]


     m = hp.read_map(m, verbose=False) if isinstance(m, str) else m
     m[ m==0. ] = np.nan # in case the input map is an apodization mask

     if min==None: min = m[ ~np.isnan(m) ].min()
     if max==None: max = m[ ~np.isnan(m) ].max()
     
     hp.orthview(m, title=title, min=min, max=max, rot=rot, half_sky=True, cmap=cmap)
     hp.graticule()

import utils
def k2d_map(map):
    alm = hp.map2alm(map)
    dlm = hp.almxfl(alm, cut * cli(q2k(np.arange(lmax + 1)))
                                     * q2d(np.arange(lmax + 1)))
                                     
    return hp.alm2map(dlm,nside)

def q2k_map(map):
    alm = hp.map2alm(map)
    klm = hp.almxfl(alm, cut * q2k(np.arange(lmax + 1)))                              
    return hp.alm2map(klm,nside)

def q2d_map(qmap):
    q2d = lambda l: (l*(l + 1)) ** 0.5 # potential -> deflection
    qlm = hp.map2alm(qmap)
    dmap = hp.alm2map(hp.almxfl(qlm, cut * q2d(np.arange(lmax + 1))),nside)
    return dmap


################################################################
############        Generate multiple tracers       ############
################################################################
def run_LSS_generator(nside,lmax,nlpp,cls):
    """
    ############     Generate the LSS tracer klms.  ############

    nside:
    lmax:
    pmap: Input lenisng potential map (signal only)
    nlpp: Reconstruction noise power spectrum, use the theory one. If multi-frequency reconstruction were used, this should be their inverse-combination noise power.
    cls:  Power spectrum matrix in the shapes of (num_of_tracers, num_of_tracers, num_of_multipoles), for example 3 tracers (k, i, g):
          the cls will be in the shapes of (3,3,lmax), and cls[i,j,:] should be the power spcreum of tracer i and tracer j.

    Returns:
    tracer_maps: ndarray
        The final tracers in the shape of (num_of_tracers)
    """

    pmap = hp.read_map(f"{P_filename}",field=(0),)
    qlm_rec = hp.read_alm(f"{rec_qlm_filename}",)
    pmap_rec = hp.alm2map(qlm_rec, nside)
    kmap_rec = q2k_map(pmap_rec)
    klm_rec = hp.map2alm(kmap_rec,lmax)

    tracer_alms = glss.ones(nside,lmax,pmap,nlpp[:lmax+1],cls)
    tracer_alms[0] = klm_rec

    np.save(f"{tracers_filename}",tracer_alms)
    return tracer_alms

tracer_alms = run_LSS_generator(nside,lmax,nlpp,cls[:,:,:lmax+1])

################################################################
############        Combine multiple tracers       ############
################################################################

def combine_weights(tracers, C_II, C_kappaI, lmax, nside, gal_cut=100):

    # 获取tracers的键名列表
    tracer_keys = list(tracers.keys())

    c_i = np.zeros((len(tracer_keys), lmax+1))

    tracer_alms = [hp.map2alm(tracers[key], lmax=lmax) for key in tracer_keys]

    combined_alm = np.zeros_like(tracer_alms[0], dtype=np.complex128)

    C_II_inv = np.zeros(C_II.shape)

    for l in range(lmax):
        try:
            C_II_inv[:,:,l] = np.linalg.inv(C_II[:,:,l])
        except:
            pass

    for i, key1 in enumerate(tracer_keys):
        for j, key2 in enumerate(tracer_keys):
            c_i[i, :] += C_kappaI[j, :] * (C_II_inv[i, j, :])

    return c_i

from generate_LSS import *
from LSS_tracer_comb import *
def run_LSS_combinatory(nside,lmax,nlpp,cls,surveys,tracer_alms):
    """

    ############     Generate the LSS tracer klms.  ############

    return: An array of klms, [combined tracer, combined tracer signal part, combined tracer noise part]
            NOTICE: the sum of noise part and signal part is the total tracer, they are used for CMB delensing debiasing by simulation.
    """

    tracers = {surveys[i]:hp.alm2map(tracer_alms[i],nside)*mask for i in range(len(surveys))}

    pmap = hp.read_map(f"{P_filename}",field=(0),)
    kmap = q2k_map(pmap)
    klm = hp.map2alm(kmap*mask,lmax)
    
    _, A = calculate_sim_weights(cls[:,:,:lmax+1])

    signals = tracers.copy()
    for i,key in enumerate(signals):
        signals[key] = hp.alm2map(hp.almxfl(klm, A[i,0,:]),nside)

    noises = tracers.copy()
    for i,key in enumerate(noises):
        noises[key] = hp.alm2map(tracer_alms[i] - hp.map2alm(signals[key],lmax),nside)

    C_II = cal_tracer_cov(nside,lmax,tracers,mask)
    C_KI = cal_kappa_tracer_cross(nside,lmax,kmap,tracers,mask)

    comb_signal = combine_maps_galcut(signals, C_II, C_KI, lmax, nside, gal_cut=100)
    comb_noise = combine_maps_galcut(noises, C_II, C_KI, lmax, nside, gal_cut=100)
    comb = combine_maps_galcut(tracers, C_II, C_KI, lmax, nside, gal_cut=100)

    comb_alms = [hp.map2alm(comb,lmax),hp.map2alm(comb_signal,lmax),hp.map2alm(comb_noise,lmax),]
    np.save(f"{comb_filename}",comb_alms)

run_LSS_combinatory(nside,lmax,nlpp,cls,surveys,tracer_alms)


def cal_rho(nside,lmax):
    """
    
    ###########  Calculate the correlation coefficient between the combined tracer and the true lensing convergence  ############

    """

    ls = np.arange(lmax+1)
    q2k = lambda l: l*(l + 1) / 2

    pmap = hp.read_map(f"{P_filename}",field=(0),)
    kmap = q2k_map(pmap)

    tracer_alms = np.load(f"{tracers_filename}")
    tracer_maps = [hp.alm2map(tracer_alms[i],nside) for i in np.arange(len(tracer_alms[:,0]))]

    comb_alms = np.load(f"{comb_filename}")
    comb_map = [hp.alm2map(comb_alms[i],nside) for i in np.arange(3)]
    clkk = hp.anafast(kmap)

    rhos = []
    for i,map in enumerate(tracer_maps):
        rhos.append(hp.anafast(kmap*mask,map*mask)[:lmax+1]/fsky/np.sqrt(clkk[:lmax+1] * hp.anafast(map*mask)[:lmax+1]/fsky))

    rho_comb = []
    for i,map in enumerate(comb_map):
        rho_comb.append(hp.anafast(kmap*mask,map*mask)[:lmax+1]/fsky/np.sqrt(clkk[:lmax+1] * hp.anafast(map*mask)[:lmax+1]/fsky))


    rhos_binned = np.zeros((len(tracer_maps), 50))
    for i in range(len(tracer_maps)):
        _,rhos_binned[i] = get_binned(nside,60,rhos[i][10:3030],lmax=3019)

    rho_comb_binned = np.zeros((len(comb_map), 50))
    for i in range(len(comb_map)):
        lbin,rho_comb_binned[i] = get_binned(nside,60,rho_comb[i][10:3030],lmax=3019)


    np.savez(f"{rho_filename}", lbin=lbin, rho_comb_binned=rho_comb_binned, rhos_binned=rhos_binned)

#cal_rho(nside,lmax)



import matplotlib.pyplot as plt
def plot_rhos():
    """
    Deprecated.
    """

    rhos0 = np.load(f"/sharefs/alicpt/users/chenwz/reconstruction_multi_process2/process_added/LSS_tracer/rhos/rhos_0.npz")
    lbin = rho0['lbin']
    rhos = rho0['rhos_binned']
    rhos_comb = rho0['rho_comb_binned']

    for i in range(1,500):
        rhosi = np.load(f"/sharefs/alicpt/users/chenwz/reconstruction_multi_process2/process_added/LSS_tracer/rhos/rhos_{i}.npz")
        rhos += rhosi['rhos_binned']
        rhos_comb += rhosi['rho_comb_binned']
    
    plt.plot(rhos[i], label='bin'+str(i))

    plt.xlim(2,3000)
    plt.ylim(0,1)
    plt.savefig(f"{rho_fig_filename}")

#plot_rhos()