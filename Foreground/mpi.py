from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


import sys

#load modules
print('loading modules...')
import numpy as np
import basic
import pickle
import curvedsky
#import cmb
import healpy as hp
import pymaster as nmt
import matplotlib.pyplot as plt

##########################################################################

import os,sys
import pylab as pl
import numpy as np
import lenspyx
import healpy as hp

sys.path.append('/sharefs/alicpt/users/chenwz/download/cmblensplus2/utils')
sys.path.append('/sharefs/alicpt/users/chenwz/Testarea/FORECAST/Foreground/run_with_foreground')

from lensingb_mine import *
from noise_making import *
from utils_mine import *
from phi_noise_making import *
from generate_map import *
from lenspyx.utils import camb_clfile
from lenspyx.utils_hp import synalm, almxfl, alm2cl
from generate_gaussian_fg import *

camb_path = os.path.realpath(os.path.join(os.getcwd(),'..'))    
sys.path.insert(0,camb_path)

import camb
from camb import model, initialpower ,correlations

from scipy.special import factorial

import pymaster as nm



# 获取传递给脚本的参数
args = sys.argv[1:]

# 打印当前作业的编号
# 计算当前作业的文件编号列表
file_indices = comm_rank + int(args[0]) #+ 200

# 打印当前作业的文件编号列表
print(f"Job {comm_rank} will process files: {file_indices}")


# map输出路径 
QUP_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/ALL_MAP_2048/QUP/QUP_{file_indices}.fits"

QU_N_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/ALL_MAP_2048/QUP_NOISE/SAT/93/QUP_N_{file_indices}.npy"

FG_filename_27 = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/ALL_MAP_2048/FG/27/IQU_FG_{file_indices}.fits"
FG_filename_39 = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/ALL_MAP_2048/FG/39/IQU_FG_{file_indices}.fits"
FG_filename_93 = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/ALL_MAP_2048/FG/93/IQU_FG_{file_indices}.fits"
FG_filename_145 = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/ALL_MAP_2048/FG/145/IQU_FG_{file_indices}.fits"
FG_filename_225 = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/ALL_MAP_2048/FG/225/IQU_FG_{file_indices}.fits"
FG_filename_280 = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/ALL_MAP_2048/FG/280/IQU_FG_{file_indices}.fits"


print(f"Job {comm_rank} will write map to {QUP_filename }:")



nside = 2048
lmax = 3*nside-1 
lmax_len = lmax # desired lmax of the lensed field.
dlmax = lmax    # lmax of the unlensed fields is lmax + dlmax.  (some buffer is required for accurate lensing at lmax)
epsilon = 1e-6  # target accuracy of the output maps (execution time has a fairly weak dependence on this)
lmax_unl, mmax_unl = lmax_len + dlmax, lmax_len + dlmax
Tcmb = 2.7255e6
npix = hp.nside2npix(nside)


#phi noise
#nlev_phi = [5.8,6.3]
#theta_ac_phi = [2.2,1.4]    #uK-arcmin
#phi_noise = gaussian_phi_n0(lmin_cut=2,lmax_cut=3*nside-1,rlmin_cut=[200],rlmax_cut=[4800],nside=nside,sigmas= nlev_phi ,thetas=theta_ac_phi,flag=5,seed = np.random.randint(99999999, size=1))


#QU noise
nlev = [10,]
theta_ac = [9,]
ac2rad = np.pi/10800.
theta_rad = np.array(theta_ac)*ac2rad

noise = homo_noise(nlev[0],nside,seed = np.random.randint(99999999, size=1))

noise *= hp.read_map('/sharefs/alicpt/users/chenwz/reconstruction_2048_28_sky/Big_bi_mask_028.fits')
noise_compressed = map_compression(map=[noise[1],noise[2]],dat=None,savepath=QU_N_filename,flag='f2d')

hp.write_map(f"{QU_N_filename}",[noise[1],noise[2]],)



"""
#FG
nus= {27: 27*1e9, 39 : 39*1e9, 93 : 93*1e9, 145 : 145*1e9, 225 : 225*1e9, 280 : 280*1e9,}
fg_map,_ = gaussian_fg(2048, nus, file_indices + 9999, same=False, method=1)
hp.write_map(f"{FG_filename_27}",fg_map[27],)
hp.write_map(f"{FG_filename_39}",fg_map[39],)
hp.write_map(f"{FG_filename_93}",fg_map[93],)
hp.write_map(f"{FG_filename_145}",fg_map[145],)
hp.write_map(f"{FG_filename_225}",fg_map[225],)
hp.write_map(f"{FG_filename_280}",fg_map[280],)
"""

"""
#CMB 
Tlen_full,Qlen_full,Ulen_full,Pmap_full = making_maps_new(nside, nrms_f=None,fwhm_f=None,phi_map=None,pixwin=False)
hp.write_map(f"{QUP_filename}",[Qlen_full,Ulen_full,Pmap_full],)
"""

#注：只有LAT的93GHz和145GHz的noise map包含QUP noise(进行delensing), SAT只有QU noise(用于cross和fit).

"""
#FG patch correction
nus= {27: 27*1e9, 39 : 39*1e9, 93 : 93*1e9, 145 : 145*1e9, 225 : 225*1e9, 280 : 280*1e9,}
for idx in np.array([14,]):
    fg_map,_ = gaussian_fg(2048, nus, idx + 24325, same=False, method=1)
    hp.write_map(f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/ALL_MAP_2048/FG/27/IQU_FG_{idx}.fits",fg_map[27],overwrite=True)
    hp.write_map(f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/ALL_MAP_2048/FG/39/IQU_FG_{idx}.fits",fg_map[39],overwrite=True)
    hp.write_map(f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/ALL_MAP_2048/FG/93/IQU_FG_{idx}.fits",fg_map[93],overwrite=True)
    hp.write_map(f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/ALL_MAP_2048/FG/145/IQU_FG_{idx}.fits",fg_map[145],overwrite=True)
    hp.write_map(f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/ALL_MAP_2048/FG/225/IQU_FG_{idx}.fits",fg_map[225],overwrite=True)
    hp.write_map(f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/ALL_MAP_2048/FG/280/IQU_FG_{idx}.fits",fg_map[280],overwrite=True)
    """