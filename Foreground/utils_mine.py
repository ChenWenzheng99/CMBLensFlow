#load modules
print('loading modules...')

import sys
sys.path.append('/sharefs/alicpt/users/chenwz/download/cmblensplus2/utils')
import numpy as np
import basic
import pickle
import curvedsky

import cmb
import healpy as hp
import pymaster as nmt
import matplotlib.pyplot as plt

##########################################################################

import os,sys
import pylab as pl
import numpy as np
import lenspyx
import healpy as hp
import plottools

from lenspyx.utils import camb_clfile
from lenspyx.utils_hp import synalm, almxfl, alm2cl

camb_path = os.path.realpath(os.path.join(os.getcwd(),'..'))    
sys.path.insert(0,camb_path)

import camb
from camb import model, initialpower ,correlations

from scipy.special import factorial



def map_compression(map=None,dat=None,savepath=None,flag='f2d'):
    """
    Given an patch sky map (.fits), we convert it to numpy array and save only non-zero pixels to a .dat file.
    OR
    Given a .dat file, we read it and convert it to a patch sky map (.fits).
    """
    def compress_array(arr):
        # Get the indices of non-zero elements
        non_zero_indices = np.nonzero(arr)
        # Get the values of the non-zero elements
        non_zero_values = arr[non_zero_indices]
        # Combine indices and values into a 2-row array
        compressed_array = np.vstack((non_zero_indices[0], non_zero_values))
        return compressed_array

    def decompress_array(compressed_array, original_shape):
        # Create an empty array of the original shape
        decompressed_array = np.zeros(original_shape)
        # Extract indices and values
        indices = compressed_array[0].astype(int)
        values = compressed_array[1]
        # Place the values back into the decompressed array
        decompressed_array[indices] = values
        return decompressed_array

    if flag == 'f2d':
        assert map is not None, "The map parameter cannot be None when flag is 'f2d'"
        map_array = []
        for i in np.arange(np.shape(map)[0]):
            map_array.append(compress_array(map[i]))
        if savepath is not None:
            np.save(savepath, map_array)
        else:
            return np.array(map_array)
        
    elif flag == 'd2f':
        assert dat is not None, "The dat parameter cannot be None when flag is 'd2f'"
        map_array = []
        for i in np.arange(np.shape(dat)[0]):
            map_array.append(decompress_array(dat[i],12*nside**2))
        if savepath is not None:
            hp.write_map(savepath, map_array)
        else:
            return np.array(map_array)


uKamin2uKpix = lambda n, npix : n / np.sqrt((360 * 60) ** 2 / np.pi / npix)  # 1° = pi/180 rad, 1' (arcmin) = pi/180/60 rad, 见PLANCK2015 (A.7)
uKpix2uKamin = lambda n, npix : n * np.sqrt((360 * 60) ** 2 / np.pi / npix)

def bl(fwhm, lmax=None, nside=None, pixwin=True):    #bl包括相当于beam window function 和 pixel window function,相当于e^-(l(l+1)* \sigma^2)(这是beam window function), 而healpy的gauss_beam函数不包含pixel window func.
    """ Transfer function.

        * fwhm      : beam fwhm in arcmin
        * lmax      : lmax
        * pixwin    : whether include pixwin in beam transfer function
        * nside     : nside
    """
    assert lmax or nside
    lmax = min( 3 * nside - 1, lmax ) if nside and lmax else lmax if lmax else 3*nside - 1
    ret = hp.gauss_beam(fwhm * np.pi / 60. / 180., lmax=lmax)   #return: beam window function
    if pixwin:
        assert nside is not None
        ret *= hp.pixwin(nside, lmax=lmax)      #hp.pixwin: Return the pixel window function for the given nside
    return ret


def bl_eft(nrms_f, fwhm_f, lmax=None, pixwin=True, ret_nlev=False):  #bl_eft 包括noise_level、beam window function 和 pixel window function,即bl乘了(\Delta_t/p)^2
    """ Effective beam.
    """
    nrms_f = [ hp.read_map(nrms) if isinstance(nrms, str) else nrms for nrms in nrms_f ] #variance map 里存的不是variance,而是standard deviation(标准差)
    nside = hp.npix2nside(len(nrms_f[0]))

    nlev_f = np.array([ uKpix2uKamin(np.mean(nrms[nrms > 0] ** -2) ** -0.5 ,     #非mask区域对应的nrms(N功率谱方均根),对应1/(nlev)^2 = np.mean(1/(nrms)^2)                                                    
                        hp.nside2npix(nside)) for nrms in nrms_f]) # in uK.radians  #这里np.mean(nrms[nrms > 0] ** -2) ** -0.5是对高斯分布求平均(先平方避免正负相消)的算法，建议使用之前画出nrms的直方图(纵轴即个数)，看看是否符合高斯分布
    nlev = sum(nlev_f ** -2) ** -0.5 # in uk.arcmin
    bl_f = [ bl(fwhm, pixwin=pixwin, lmax=lmax, nside=nside) for fwhm in fwhm_f ]
    bl_eft = (sum([ nlev ** -2 * bl ** 2 for nlev, bl in zip(nlev_f, bl_f) ])
                * nlev ** 2) ** 0.5
    
    
    if ret_nlev:
        return nlev, bl_eft  #返回noiselevel和Effective beam
    else:
        return bl_eft        #仅返回Effective beam


def cli(cl):
    """Pseudo-inverse for positive cl-arrays.

    """
    ret = np.zeros_like(cl)
    ret[np.where(cl != 0)] = 1. / cl[np.where(cl != 0)]
    return ret


def lm_healpix2healpy(alm2d, lmax=None):
    # get the lmax
    if lmax is None:
        nl = len(alm2d[0]) # this is number of l = lmax+1
    else:
        nl = lmax + 1
        # lmax should < len(alm2d[0]) 
    # print(lmax)
    # init
    alm = []

    # for each m
    for l in range(nl):
        alm.extend(alm2d[l:nl, l])

    return np.array(alm)


def get_binned(nside,nbl,cl,lmax=0):   
    import numpy as np
    import matplotlib.pyplot as plt
    import pymaster as nmt

    # HEALPix 地图分辨率
    #nside = 256

    # 将功率谱分成   个 bin，每个 bin 有 nbl 个多极项
    if lmax != 0:
        bin1 = nmt.NmtBin.from_lmax_linear(lmax, nbl)
    else: 
        bin1 = nmt.NmtBin.from_nside_linear(nside, nbl)

    # 生成全是 1 的 C_\ell^{TT}
    #cl_tt = dlbb_de1[0:len(L)]-dlbb_noise_com[0:len(L)]-dlbb_noise10

    # 将功率谱分 bin 成 bandpowers
    cl_tt_binned = bin1.bin_cell(np.array([cl]))

    # 有效多极项的数组
    ell_eff = bin1.get_effective_ells()

    # 绘制结果，使用点图而不是折线图
    #plt.plot(ell_eff, cl_tt_binned[0], 'g.', label='bin  $C_\\ell^{TT}$')
    #plt.loglog()
    #plt.legend(loc='upper right', frameon=False)
    #plt.show()
    return ell_eff,cl_tt_binned[0]

def calc_fsky(nside,masks,mask): # fsky calculation
    pixarea=hp.nside2pixarea(nside)
    #print(pixarea)
    ret2 = np.ones_like(masks)
    ret4 = np.ones_like(masks)
    ret2 *= masks**2
    ret4 *= masks**4
    order2=np.sum(ret2)
    order4=np.sum(ret4)
    fsky=pixarea/4/np.pi        #fsky
    fsky=fsky*order2**2/order4  #order2==order4???
    return fsky

def get_pseudo_cl(nside,mask_input,flag,bin,lmax=None,apo=False,f0=None,f2=None,is_Dell=True):
    """
        nside:
        mask_input:最好是1024的mask,否则apodization很慢;或者输入apodization后的mask
        flag: '00' '02' '22',依次为spin-0 x spin-0,spin-0 x spin-2,spin-2 x spin-2
        bin: 每个bin的多极项数目
        lmax: 用于计算pseudo-Cl的lmax,默认为-1,即使用3*nside-1
        apo: 是否输入了apodization后的mask
        f0: spin-0 field
        f2: spin-2 field
    """
        
    import numpy as np
    import healpy as hp
    import matplotlib.pyplot as plt

    # Import the NaMaster python wrapper
    import pymaster as nmt

    if lmax is None:
        lmax = 3*nside - 1
    if apo == True:
        mask = mask_input
    else:
        mask = nmt.mask_apodization(mask_input,  1., apotype="Smooth")           # apodization中的参数可以调整，这里为 1 degree
    #hp.orthview(mask, half_sky=True, rot=[180, 30, 0],cmap='YlGnBu_r',title='Apodized mask')
    print('mask apodization done')
    
    # Read healpix maps and initialize a spin-0 and spin-2 field
    
    # Initialize binning scheme with 'bin' ells per bandpower
    #b = nmt.NmtBin.from_nside_linear(nside, bin, is_Dell=is_Dell)
    b = nmt.NmtBin.from_lmax_linear(lmax, bin, is_Dell=is_Dell)
    ell_arr = b.get_effective_ells()

    # Compute MASTER estimator
    # spin-0 x spin-0
    if flag == '00':
        f_0 = nmt.NmtField(mask, [f0],)
        assert f_0 is not None
        print('pseudo cl calculating ')
        cl_00 = nmt.compute_full_master(f_0, f_0, b)    #TT为 cl_00[0]
        return ell_arr,cl_00

    # spin-0 x spin-2
    if flag == '02':
        f_0 = nmt.NmtField(mask, [f0],)
        f_2 = nmt.NmtField(mask, f2, spin=2, purify_e=True, purify_b=True, n_iter_mask_purify=3, )
        assert f_0 is not None
        assert f_2 is not None
        print('pseudo cl calculating ')
        cl_02 = nmt.compute_full_master(f_0, f_2, b)
        return ell_arr,cl_02
    
    # spin-2 x spin-2
    if flag == '22':
        f_2 = nmt.NmtField(mask, f2, spin=2, purify_e=True, purify_b=True, n_iter_mask_purify=3, )
        assert f_2 is not None
        print('pseudo cl calculating ')
        cl_22 = nmt.compute_full_master(f_2, f_2, b)      #EE,BB 分别为 cl_22[0]，cl_22[3]
        return ell_arr,cl_22



def E2Bcorrection(nside,bin,map,binary_mask,lmax=None,flag='clean',n_iter=3, is_Dell=True):
    """
        nside
        lmax: 用于计算pseudo-Cl的lmax
        bin: 每个bin的多极项数目
        map: TQU maps with mask on
        binary_mask: 0-1二值掩模
        n_iter: EB leakage correction的迭代次数
    """
    from eblc_base import EBLeakageCorrection

    import numpy as np
    import healpy as hp
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    from pathlib import Path
    import glob
    import pymaster as nmt

    if lmax is None:
        lmax = 3*nside - 1
    obj = EBLeakageCorrection(map, lmax=lmax, nside=nside, mask=binary_mask, post_mask=binary_mask, method='cutqufitqu', check_res=False, n_iter=n_iter)
    crt_b, tmp_b, cln_b = obj.run_eblc()

    #b = nmt.NmtBin.from_nside_linear(nside, bin, is_Dell=is_Dell)
    b = nmt.NmtBin.from_lmax_linear(lmax, bin, is_Dell=is_Dell)
    ell_arr = b.get_effective_ells()

    if flag == 'clean' :
        f_0_clean = nmt.NmtField(binary_mask, [cln_b], )
        cl_00_clean = nmt.compute_full_master(f_0_clean, f_0_clean, b)
        return cln_b,ell_arr,cl_00_clean[0]

    if flag == 'corrupt':
        f_0_corrupt = nmt.NmtField(binary_mask, [crt_b], )
        cl_00_corrupt = nmt.compute_full_master(f_0_corrupt, f_0_corrupt, b)
        return crt_b,ell_arr,cl_00_corrupt[0]

    if flag == 'template':
        f_0_template = nmt.NmtField(binary_mask, [tmp_b], )
        cl_00_template = nmt.compute_full_master(f_0_template, f_0_template, b)
        return tmp_b,ell_arr,cl_00_template[0]


def alm2map_1024(alm, lmax=None):
    """
    Convert 2048 alm to 1024 map.(for partial sky QU only)
    """
    nside_cut = 1024
    if lmax is None:
        lcut = 3*nside_cut - 1
    else:
        lcut = lmax
    Q,U = curvedsky.utils.hp_alm2map_spin(nside_cut, lcut, lcut, 2, alm[1][:lcut+1,:lcut+1], alm[2][:lcut+1,:lcut+1])
    return Q,U


def alm2map_degrade(nside_cut, alm, lmax=None):
    """
    Convert higher alm to lower map.
    For TQU only, multipoles of alm should >= lmax and 3*nside_cut-1
    alm: tlm,elm,blm
    """
    nside_cut = nside_cut
    if lmax is None:
        lcut = 3*nside_cut - 1
    else:
        lcut = lmax
    T = curvedsky.utils.hp_alm2map(nside_cut, lcut, lcut, alm[0][:lcut+1,:lcut+1])
    Q,U = curvedsky.utils.hp_alm2map_spin(nside_cut, lcut, lcut, 2, alm[1][:lcut+1,:lcut+1], alm[2][:lcut+1,:lcut+1])
    return Q,U


def map2cl_1024(Q,U,mask_1024,bin,apo=False):
    """
    Convert 1024 Q/U map to pseudo-Cl (EE为cl[0],BB为cl[3]).(for partial sky QU only)
    """

    l,cl = get_pseudo_cl(1024,mask_1024,'22',bin,apo,f2=[Q,U])
    return l,cl


def map_degrade(nside_in,nside_out,map_in,lmax_in=None,lmax_out=None):
    """
    Degrade 2048 map to 1024 map。该做法在l<2000范围内安全(功率谱重合). map_in请按IQU的顺序输入。
    """
    if lmax_in is None:
        lmax = 3*nside_in - 1
    else:
        lmax = lmax_in

    tlm = curvedsky.utils.hp_map2alm(nside_in, lmax, lmax, map_in[0])       #2D unlensed alm
    elm,blm = curvedsky.utils.hp_map2alm_spin(nside_in, lmax, lmax, 2, map_in[1], map_in[2])       #2D lensed alm

    if lmax_out is None:
        lcut = 3*nside_out - 1
    else:
        lcut = lmax_out
    Tmap_1024 = curvedsky.utils.hp_alm2map(nside_out, lcut, lcut, tlm[:lcut+1,:lcut+1])       #2D unlensed map
    Qmap_1024,Umap_1024 = curvedsky.utils.hp_alm2map_spin(nside_out, lcut, lcut, 2, elm[:lcut+1,:lcut+1], blm[:lcut+1,:lcut+1])

    map_out = np.array([Tmap_1024,Qmap_1024,Umap_1024])
    return map_out





def Wiener_filter_QU(nside,lmax,noisy_maps,noise,):
    elm_full_noisy,_ = curvedsky.utils.hp_map2alm_spin(nside, lmax, lmax, 2, noisy_maps[1], noisy_maps[2])       
    clee_len_full_noisy = curvedsky.utils.alm2cl(lmax, elm_full_noisy, alm2=None)

    elm_noise,_ = curvedsky.utils.hp_map2alm_spin(nside, lmax, lmax, 2, noise[1], noise[2])
    clee_noise = curvedsky.utils.alm2cl(lmax, elm_noise, alm2=None)

    ratio = (clee_len_full_noisy - clee_noise) * cli(clee_len_full_noisy)
    ratio[1]=ratio[2]
    return ratio

def Wiener_filter_phi(nside,lmax,pure_map,noisy_map,):
    plm_pure = curvedsky.utils.hp_map2alm(nside, lmax, lmax, pure_map)
    plm_noisy_full = curvedsky.utils.hp_map2alm(nside, lmax, lmax, noisy_map)
    clpp_cross_full = curvedsky.utils.alm2cl(lmax, plm_pure, alm2=plm_noisy_full)
    clpp_noisy_full = curvedsky.utils.alm2cl(lmax, plm_noisy_full, alm2=None)
    ratio = (clpp_cross_full) * cli(clpp_noisy_full)
    return ratio

def QU2comb(nside,lmax,Qlm,Ulm):
    Qlen_fil = hp.alm2map(Qlm, nside, )        #combined & filtered noisy lensed QU map
    Ulen_fil = hp.alm2map(Ulm, nside, )
    elm_fil_pix,blm_fil_pix = curvedsky.utils.hp_map2alm_spin(nside, lmax,lmax, 2, Qlen_fil, Ulen_fil)       #2D lensed alm
    tlm_fil_pix = elm_fil_pix
    combined_alm_len = np.stack((tlm_fil_pix, elm_fil_pix, blm_fil_pix), axis=0)
    return combined_alm_len

def alm2cl(alm,mask_apo,lmax=None,is_Dell=True):
    """
    calculate pseudo-Cl from partial sky alm
    alm: [T,Q,U] or [Q,Q,U]
    lmax: lmax of pseudo-Cl, default is 3 * 1024 - 1
    mask_apo: had better apodized mask
    """
    
    nside_cut = 1024
    lcut = 3*nside_cut - 1
    Q,U = curvedsky.utils.hp_alm2map_spin(nside_cut, lcut, lcut, 2, alm[1][:lcut+1,:lcut+1], alm[2][:lcut+1,:lcut+1])
    if mask_apo is not None:
        Q *= mask_apo
        U *= mask_apo
    if lmax is None:
        lmax = 3*nside_cut - 1
    _,l1,cl_de = E2Bcorrection(1024,bin,[Q,Q,U,],mask_apo,lmax=lmax,flag='clean',n_iter=3, is_Dell=is_Dell)
    return l1,cl_de

def cutting_lrange(alm,lmax,l_low=0,l_top=99999):
    """
        l_low: lower bound of l range after cut
        l_top: upper bound of l range after cut

    """

    alm_pix = curvedsky.utils.lm_healpy2healpix(alm, lmax, lmpy=0)

    alm_pix[:l_low,:] = 0
    alm_pix[l_top + 1:,:] = 0

    alm_py = lm_healpix2healpy(alm_pix, )

    return alm_py