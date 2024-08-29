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
file_indices = comm_rank + int(args[0])

# 打印当前作业的文件编号列表
print(f"Job {comm_rank} will process files: {file_indices}")

# map读取路径 /map_TQU_2048_0000.fits
rec_qlm_filename = f"/sharefs/alicpt/users/chenwz/reconstruction_multi_process2/process_added/com_rec/rec_qlm_{file_indices}.fits"
P_filename = f"/sharefs/alicpt/users/chenwz/reconstruction_2048_simons/test/ALILENS/sims/cmbs/map_P_2048_{file_indices:04d}.fits"

nlpp = np.loadtxt('/sharefs/alicpt/users/chenwz/reconstruction_2048_simons/wiener.dat')[1]

CIB_filename = f"/sharefs/alicpt/users/chenwz/reconstruction_multi_process2/process_added/LSS_tracer/CIB/CIB_{file_indices}.fits"
Euclid_filename = f"/sharefs/alicpt/users/chenwz/reconstruction_multi_process2/process_added/LSS_tracer/Euclid/Euclid_{file_indices}.fits"
LSST_filename = f"/sharefs/alicpt/users/chenwz/reconstruction_multi_process2/process_added/LSS_tracer/LSST/LSST_{file_indices}.fits"
WFIRST_filename = f"/sharefs/alicpt/users/chenwz/reconstruction_multi_process2/process_added/LSS_tracer/WFIRST/WFIRST_{file_indices}.fits"
SKA_filename = f"/sharefs/alicpt/users/chenwz/reconstruction_multi_process2/process_added/LSS_tracer/SKA/SKA_{file_indices}.fits"

rho_filename = f"/sharefs/alicpt/users/chenwz/reconstruction_multi_process2/process_added/LSS_tracer/rhos/rhos_{file_indices}.txt"
rho_fig_filename = f"/sharefs/alicpt/users/chenwz/reconstruction_multi_process2/process_added/LSS_tracer/rhos_fig/rhos_fig_{file_indices}.png"

nside = 2048
lmax = 3*1024-1
cls = np.load('/sharefs/alicpt/users/chenwz/reconstruction_multi_process2/process_added/LSS_power_matrix.npy')

def run_LSS_generator(nside,lmax,nlpp,cls):
    """
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

    tracer_alms = glss.ones(nside,lmax,pmap,nlpp[:lmax+1],cls)

    hp.write_alm(f'{CIB_filename}',tracer_alms[1],overwrite=True)
    hp.write_alm(f'{Euclid_filename}',tracer_alms[2],overwrite=True)
    hp.write_alm(f'{LSST_filename}',tracer_alms[3],overwrite=True)
    hp.write_alm(f'{WFIRST_filename}',tracer_alms[4],overwrite=True)
    hp.write_alm(f'{SKA_filename}',tracer_alms[5],overwrite=True)

#run_LSS_generator(nside,lmax,nlpp,cls[:,:,:lmax+1])


mask = hp.read_map('/sharefs/alicpt/users/chenwz/Testarea/FORECAST/mask/AliCPT_20uKcut150_C_2048.fits')
fsky = calc_fsky(nside,mask,mask)

def cal_rho(nside,lmax):
    pmap = hp.read_map(f"{P_filename}",field=(0),)
    plm = hp.map2alm(pmap, lmax)

    plm_rec = hp.read_alm(f"{rec_qlm_filename}")

    ls = np.arange(lmax+1)
    q2k = lambda l: l*(l + 1) / 2
    klm = hp.almxfl(plm, q2k(ls))
    klm_rec = hp.almxfl(plm_rec, q2k(ls))

    kmap = hp.alm2map(klm,nside)
    kmap_rec = hp.alm2map(klm_rec,nside)
    CIB_map = hp.alm2map(hp.read_alm(f"{CIB_filename}"),nside)
    Euclid_map = hp.alm2map(hp.read_alm(f"{Euclid_filename}"),nside)
    LSST_map = hp.alm2map(hp.read_alm(f"{LSST_filename}"),nside)
    WFIRST_map = hp.alm2map(hp.read_alm(f"{WFIRST_filename}"),nside)
    SKA_map = hp.alm2map(hp.read_alm(f"{SKA_filename}"),nside)

    maps = [kmap_rec,CIB_map,Euclid_map,LSST_map,WFIRST_map,SKA_map]

    clkk = hp.anafast(kmap)

    rhos = []
    for i,map in enumerate(maps):
        rhos.append(hp.anafast(kmap*mask,map*mask)/fsky/np.sqrt(clkk * hp.anafast(map*mask)/fsky))

    np.savetxt(f"{rho_filename}",rhos)

#cal_rho(nside,lmax)



import matplotlib.pyplot as plt
def plot_rhos():
    rhos = np.loadtxt(f"{rho_filename}")
    for i in range(len(rhos)):
        plt.plot(rhos[i], label='bin'+str(i))

    plt.xlim(2,3000)
    plt.ylim(0,1)
    plt.savefig(f"{rho_fig_filename}")

#plot_rhos()