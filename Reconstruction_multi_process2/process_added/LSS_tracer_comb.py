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
sys.path.append('/root/Testarea/prototype/Foreground')
from lensingb_mine import *
from noise_making import *
from utils_mine import *
from phi_noise_making import *
from generate_map import *
from lenspyx.utils import camb_clfile
from lenspyx.utils_hp import synalm, almxfl, alm2cl

camb_path = os.path.realpath(os.path.join(os.getcwd(),'..'))    
sys.path.insert(0,camb_path)

import camb
from camb import model, initialpower ,correlations

from scipy.special import factorial

import pymaster as nm


"""

Multiple tracers are combined with coeficeints, which is derived when we maximize the correlation coefficient between the combined tracer and the true lensing convergence.
This combination coefficient is something like a Wiener-filter, so the combined tracer is actually the filtered one.

"""


def cal_tracer_cov(nside,lmax,tracers,mask):
    """
    Calculate the covmat of the tracer maps, C_{II}

    tracers: A dict of tracer maps with or without masks on,
                such as : tracers = {'kappa':tracer_maps[0]*mask, 'cib':tracer_maps[1]*mask, 'Euclid': tracer_maps[2]*mask, 'LSST': tracer_maps[3]*mask, 'WFIRST': tracer_maps[4]*mask, 'SKA': tracer_maps[5]*mask,}
    mask: The mask applied to the tracer maps
    """
    fsky = calc_fsky(nside,mask,mask)

    # Ensure the same mask on the tracer maps
    for key in tracers.keys():
        tracers[key] = tracers[key]*mask

    # 获取tracers的键名列表
    tracers_keys = list(tracers.keys())
    tracer_num = len(tracers_keys)

    # 初始化一个6x6的结果数组
    result_matrix = np.zeros((tracer_num, tracer_num, lmax+1))

    # 计算不同tracer组合之间的cal_LSS_power
    for i, key1 in enumerate(tracers_keys):
        for j, key2 in enumerate(tracers_keys):
            result = hp.anafast(tracers[key1],tracers[key2],lmax=lmax)/fsky
            result_matrix[i, j, :] = result
    return result_matrix

def cal_kappa_tracer_cross(nside,lmax,kmap,tracers,mask):
    """
    Calculate the cross-spectrum between true kappa and the tracer maps, C_{\kappa I}

    tracers: A dict of tracer maps with or without masks on,
                such as : tracers = {'kappa':tracer_maps[0]*mask, 'cib':tracer_maps[1]*mask, 'Euclid': tracer_maps[2]*mask, 'LSST': tracer_maps[3]*mask, 'WFIRST': tracer_maps[4]*mask, 'SKA': tracer_maps[5]*mask,}
    mask: The mask applied to the tracer maps
    """
    fsky = calc_fsky(nside,mask,mask)

    # Ensure the same mask on the tracer maps
    for key in tracers.keys():
        tracers[key] = tracers[key]*mask
    kmap *= mask

    # 获取tracers的键名列表
    tracers_keys = list(tracers.keys())
    tracer_num = len(tracers_keys)

    # 初始化一个6x1的结果数组
    result_matrix2 = np.zeros((tracer_num, lmax+1))

    # 计算不同tracer组合之间的cal_LSS_power
    for i, key1 in enumerate(tracers_keys):
            result = hp.anafast(kmap,tracers[key1],lmax=lmax) /fsky
            result_matrix2[i, :] = result
    return result_matrix2



import numpy as np
import healpy as hp

def combine_maps(tracers, C_II, C_kappaI, lmax, nside, gal_cut=100):
    """
    根据公式 I = Σ c_i * I_i 组合 tracer maps, 首先将 map 转换为 alm。

    参数:
    maps: numpy.ndarray
        形状为 (6, N) 的二维数组，其中每一行是一个 tracer 的 map，N 是每个 map 的像素数。
    C_II: numpy.ndarray
        形状为 (6, 6, lmax+1) 的三维数组，表示每个多极矩的协方差矩阵。
    C_kappaI: numpy.ndarray
        形状为 (6, lmax+1) 的二维数组，表示每个多极矩下的 C_{kappa I_j}。
    lmax: int
        要计算的最大多极矩数。

    返回:
    I_map: numpy.ndarray
        组合后的 map, 长度为 N 的一维数组。
    """
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
            
        #if i==0:            
            #c_i[i, :gal_cut] = 1
        #else:
            #c_i[i, :gal_cut] = 0   # Cut CIB and galaxy tracers at l<200, to exclude large galactic foreground
    
        
    for i, key1 in enumerate(tracer_keys):
            combined_alm += hp.almxfl(tracer_alms[i], c_i[i,:])




    # 将组合后的 alm 转换为 map
    I_map = hp.alm2map(combined_alm, nside)

    return I_map

def combine_maps_galcut(tracers, C_II, C_kappaI, lmax, nside, gal_cut=100):
    """
    根据公式 I = Σ c_i * I_i 组合 tracer maps, 首先将 map 转换为 alm。

    参数:
    maps: numpy.ndarray
        形状为 (6, N) 的二维数组，其中每一行是一个 tracer 的 map，N 是每个 map 的像素数。
    C_II: numpy.ndarray
        形状为 (6, 6, lmax+1) 的三维数组，表示每个多极矩的协方差矩阵。
    C_kappaI: numpy.ndarray
        形状为 (6, lmax+1) 的二维数组，表示每个多极矩下的 C_{kappa I_j}。
    lmax: int
        要计算的最大多极矩数。

    返回:
    I_map: numpy.ndarray
        组合后的 map, 长度为 N 的一维数组。
    """
    def remove_row_and_column(A, i, j):
        # Delete the i-th row
        A_new = np.delete(A, i, axis=0)
        # Delete the j-th column from the remaining matrix
        B = np.delete(A_new, j, axis=1)
        return B


    # 获取tracers的键名列表
    tracer_keys = list(tracers.keys())
    c_i = np.zeros((len(tracer_keys), lmax+1))
    tracer_alms = [hp.map2alm(tracers[key], lmax=lmax) for key in tracer_keys]
    combined_alm = np.zeros_like(tracer_alms[0], dtype=np.complex128)
    C_II_inv = np.zeros(C_II.shape)


    # Create a new tracer dict to exclude the CIB 
    tracers_galcut = tracers.copy()
    del tracers_galcut[tracer_keys[1]]    # 删除新字典中的CIB对应的键值对（索引为1）
    tracers_galcut_keys = list(tracers_galcut.keys())
    tracer_galcut_alms = np.delete(tracer_alms, 1, axis=0)
    combined_alm2 = np.zeros_like(tracer_alms[0], dtype=np.complex128)
    C_II_galcut = remove_row_and_column(C_II, 1, 1)
    C_II_inv_galcut = np.zeros(C_II_galcut.shape)
    C_kappaI_galcut = np.delete(C_kappaI, 1, axis=0)
    c_i_galcut = np.zeros((len(tracers_galcut_keys), lmax+1))

    def cal_and_comb(C_II_f,C_II_inv_f,tracer_keys_f,C_kappaI_f,c_i_f,tracer_alms_f,combined_alm_f):
        for l in range(lmax):
            try:
                C_II_inv_f[:,:,l] = np.linalg.inv(C_II_f[:,:,l])
            except:
                pass
        
        for i, key1 in enumerate(tracer_keys_f):
            for j, key2 in enumerate(tracer_keys_f):
                c_i_f[i, :] += C_kappaI_f[j, :] * (C_II_inv_f[i, j, :])
                
        for i, key1 in enumerate(tracer_keys_f):
                combined_alm_f += hp.almxfl(tracer_alms_f[i], c_i_f[i,:])
        return combined_alm_f
    
    combined_alm = cal_and_comb(C_II, C_II_inv, tracer_keys, C_kappaI, c_i, tracer_alms, combined_alm)
    combined_alm2 = cal_and_comb(C_II_galcut, C_II_inv_galcut, tracers_galcut_keys, C_kappaI_galcut, c_i_galcut, tracer_galcut_alms, combined_alm2)


    # Combined two l range alms (l<100 & l>100)
    cut = np.where( (np.arange(lmax + 1) < gal_cut), 0, 1) # band limit

    combined_final_alm = hp.almxfl(combined_alm, cut) + hp.almxfl(combined_alm2, 1-cut)

    # 将组合后的 alm 转换为 map
    I_map = hp.alm2map(combined_final_alm, nside)

    return I_map