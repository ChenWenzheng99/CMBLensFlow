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
sys.path.append('/sharefs/alicpt/users/chenwz/Testarea/FORECAST/Foreground/foreground_run')

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


sys.path.append("/root/Testarea/prototype/Foreground")
from utils_mine import *


import pymaster as nmt

def get_pseudo_cl(nside,mask_input,flag,bin,lmax=None,apo=False,f0=None,f2=None,is_Dell=True):
    """
        nside:
        mask_input:最好是nside的mask,否则apodization很慢;或者输入apodization后的mask
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
    if isinstance(f0, (list, tuple)):
        f0_1 = [f0[0]]
        f0_2 = [f0[1]]
    else:
        f0_1 = [f0]
        f0_2 = [f0]
    print(len(f0_1),len(f0_2))
    # Compute MASTER estimator
    # spin-0 x spin-0
    if flag == '00':
        f_01 = nmt.NmtField(mask, f0_1,)
        f_02 = nmt.NmtField(mask, f0_2,)

        print('pseudo cl calculating ')
        cl_00 = nmt.compute_full_master(f_01, f_02, b)    #TT为 cl_00[0]
        return ell_arr,cl_00

    # spin-0 x spin-2
    if flag == '02':
        f_0 = nmt.NmtField(mask, f0_1,)
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
    
def add_beam(fwhm_ac,nside,maps,lmax=None,pixwin=False):
    if lmax == None:
        lmax =3*nside-1
    transf = bl(fwhm_ac, nside=nside, lmax=lmax, pixwin=pixwin)
    tlm_len = hp.map2alm(maps[0], lmax=lmax)
    elm_len, blm_len = hp.map2alm_spin([maps[1], maps[2]], 2, lmax=lmax)
    Tlen = hp.alm2map(hp.almxfl(tlm_len, transf, inplace=True), nside, verbose=False)
    Qlen, Ulen = hp.alm2map_spin([hp.almxfl(elm_len, transf), hp.almxfl(blm_len, transf)], nside, 2, lmax)
    return Tlen,Qlen,Ulen

def map_cut(nside,maps,lmax):
    tlm_len = hp.map2alm(maps[0], lmax=lmax)
    elm_len, blm_len = hp.map2alm_spin([maps[1], maps[2]], 2, lmax=lmax)
    Tlen = hp.alm2map(tlm_len,nside, verbose=False)
    Qlen, Ulen = hp.alm2map_spin([elm_len, blm_len,], nside, 2, lmax)
    return Tlen,Qlen,Ulen


def decon_map(fwhm_ac,nside,maps,lmax,pixwin=False):
    """
    deconvolve map with beam
    """
    if lmax == None:
        lmax =3*nside-1
    transf = 1/bl(fwhm_ac, nside=nside, lmax=lmax, pixwin=pixwin)
    tlm_len = hp.map2alm(maps[0], lmax=lmax)
    elm_len, blm_len = hp.map2alm_spin([maps[1], maps[2]], 2, lmax=lmax)
    Tlen = hp.alm2map(hp.almxfl(tlm_len, transf, inplace=True), nside, verbose=False)
    Qlen, Ulen = hp.alm2map_spin([hp.almxfl(elm_len, transf), hp.almxfl(blm_len, transf)], nside, 2, lmax)
    return Tlen,Qlen,Ulen

def qu2eb_map(nside,lmax,Qmap,Umap):
    """convert Q/U map to E/B map, only correct for full sky, but leads negligible corruption to partial sky E mode"""
    elm,blm = hp.map2alm_spin([Qmap,Umap],2)
    Emap = hp.alm2map(elm,nside,)
    Bmap = hp.alm2map(blm,nside,)
    return Emap,Bmap


#Define a function to get the theory Cl
print('setting parameters...')
nside = 1024
lmax = 3*nside-1
lmax_len = 3*nside-1 # desired lmax of the lensed field.
dlmax = 3*nside-1   # lmax of the unlensed fields is lmax + dlmax.  (some buffer is required for accurate lensing at lmax)
epsilon = 1e-6  # target accuracy of the output maps (execution time has a fairly weak dependence on this)
lmax_unl, mmax_unl = lmax_len + dlmax  , lmax_len + dlmax 
if nside == 2048 or nside == 4096:
    lmax_unl, mmax_unl = 10000  , 10000
Tcmb = 2.7255e6

#Define a function to get the theory Cl
def get_cl(r):
    #temporary used for generating the input Cls
    #运行Camb，给出理论计算结果
    #print('running CAMB...')
    #Set up a new set of parameters for CAMB
    #对象 pars 用于存储和设置 CAMB 的参数
    pars = camb.CAMBparams()
    #This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency

    #parameters from planck FFP10_wdipole_params.ini
    #通过调用 set_cosmology 方法设置宇宙学参数。其中： H0为Hubble 常数、ombh2为重子物质密度参数、omch2为冷暗物质密度参数、mnu为中微子的质量、omk为曲率参数 、tau为再电离光深度 
    pars = camb.set_params(H0=67.01904, ombh2=0.02216571, omch2=0.1202944, mnu=0.0006451439, omk=0, tau=0.06018107, YHe=0.2453006, num_nu_massless=2.03066666667, nu_mass_eigenstates = 1,\
                        num_nu_massive=1, share_delta_neff=False, nu_mass_degeneracies = [1.01533333333],nu_mass_fractions = [1], Alens = 1.00000000000000,\
                        w=-1.0, wa=0, nnu=3.046, nt=-r/8.0, nrun=0, pivot_scalar=0.05, pivot_tensor=0.05, TCMB = 2.7255 )

    # 启用张量模式
    pars.WantTensors = True

    #使用 set_params 方法设置功率谱的初始参数，包括幂律指数 ns、标量扰动的振幅 As 和张量-标量比例 r。
    pars.InitPower.set_params(As=2.119631e-09, ns=0.9636852, r=r)

    #使用 set_for_lmax 方法设置计算的最大多极限 lmax 和透镜势计算的精度 lens_potential_accuracy。
    pars.set_for_lmax(lmax_unl, lens_potential_accuracy=8)

    #calculate results for these parameters
    results = camb.get_results(pars)

    ##############################################################################
    #生成时无量纲，对CMB auto spectrum 要乘Tcmb**2, 对lensing potential auto spectrum 保持无量纲, 对CMB-potential cross spectrum 乘Tcmb
    cl_ppte=results.get_lens_potential_cls(lmax=None, CMB_unit='muK', raw_cl=True)   #numpy array CL[0:lmax+1,0:3], where 0..2 indexes PP, PT, PE.(T,E均unlensed)
    clpp=cl_ppte[:,0]

    unlensed_total = results.get_unlensed_total_cls(raw_cl=False) * Tcmb**2 # unlensed CMB power spectra, including tensors

    dl_ppte=results.get_lens_potential_cls(lmax=None, CMB_unit='muK', raw_cl=False)   #numpy array CL[0:lmax+1,0:3], where 0..2 indexes PP, PT, PE.(T,E均unlensed)
    dlpp=dl_ppte[:,0]
    lensed_cl = results.get_lensed_cls_with_spectrum(dlpp, lmax=None, CMB_unit=None, raw_cl=False) * Tcmb**2
    return unlensed_total,lensed_cl

#NOTICE: Here we use uK_CMB unit.
class fg_model:
    def __init__(self):
        None
    def unit_convert(self,nu):  #NOTICE: Both of our CMB map and fg map are in KCMB unit, therefore we don't need this unit conversion, and work just in KCMB unit.
        h = 6.62607015e-34
        k = 1.380649e-23
        Tcmb = 2.7255
        x = h*nu/k/Tcmb
        f_CMB2RJ = np.exp(x)*(x/(np.exp(x)-1))**2   
        return f_CMB2RJ
    
    def fg_spectra(self, nu, beta_s, beta_d, Td = 19.6, nu0_s=23*1e9, nu0_d=353*1e9):  #NOTICE :Since our fg map are in KCMB unit, this fg_spectra should also be converted to KCMB unit.
        """
        A factor concerning frequency dependence of the fg spatial spectra.
        """
        h = 6.62607015e-34
        k = 1.380649e-23
        Tcmb = 2.7255
        Td = 19.6   #in K
        f_s = (nu/nu0_s) ** beta_s     / (self.unit_convert(nu)/self.unit_convert(nu0_s))                           #If you use uKCMB unit, you should add this unit conversion.
        f_d = (nu/nu0_d) ** (beta_d + 1) * (np.exp(h*nu0_d/k/Td)-1) / (np.exp(h*nu/k/Td)-1)    / (self.unit_convert(nu)/self.unit_convert(nu0_d))
        return f_s, f_d
    
    def decorrelation(self, delta_s,delta_d,  nu1, nu2, rat, nu0_s=[23*1e9, 33*1e9], nu0_d=[217*1e9, 353*1e9,]):
        # Calculate factor by which foreground (dust or sync) power is decreased
        # for a cross-spectrum between two different frequencies.
        # 暂时不用这个函数，若使用需要根据nuo_d和nuo_s的D_l确定delta_s和delta_d的值，参考BICEP2: arcxiv:1810.05216的附录H和G。
        if nu1 == nu2:
            decorr_s = 1
            decorr_d = 1
        
        decorr_ss = delta_s**( (np.log(nu1 / nu2) ** 2) / (np.log(nu0_s[0] / nu0_s[1]) ** 2) )
        decorr_dd = delta_d**( (np.log(nu1 / nu2) ** 2) / (np.log(nu0_d[0] / nu0_d[1]) ** 2) )

        return decorr_ss, decorr_dd
        
    def Dl_fg_nu(self, lmax, bin, A, alpha, l0 = 80, if_bin = True):  # We only consdier BB power spectrum here. 
        """
        Deprecated!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """
        l = np.arange(10000)
        Dl = A * (l/l0) ** alpha                            #This is the power law Dl of the fg in unit of uK_RJ^2.
        if if_bin == 1:
            _, output = get_binned(nside,bin,Dl[:lmax+1],lmax)
        else:
            output = Dl
        return output
    
    def Dl_fg_auto_nu(self, nu1, nu2, lmax, bin, A_d, A_s, alpha_s, alpha_d, beta_s, beta_d, l0 = 80, if_bin = True):  #Retuen Dl_dd, Dl_ss for single frequency
        l = np.arange(10000)
        Dl_d = A_d * (l/l0) ** alpha_d
        Dl_s = A_s * (l/l0) ** alpha_s
        f_s1,f_d1 = self.fg_spectra(nu1, beta_s, beta_d)
        f_s2,f_d2 = self.fg_spectra(nu2, beta_s, beta_d)
        Dl_dd = f_d1 *f_d2 * Dl_d
        Dl_ss = f_s1 *f_s2 * Dl_s
        if if_bin == 1:
            lbin, output1 = get_binned(nside,bin,Dl_dd[:lmax+1],lmax)
            lbin, output2 = get_binned(nside,bin,Dl_ss[:lmax+1],lmax)
        else:
            lbin, output1 = l,Dl_dd
            lbin, output2 = l,Dl_ss
        return lbin,output1,output2

    def Dl_fg_cross_nu(self, nu1, nu2, lmax, bin, A_d, A_s, alpha_s, alpha_d, beta_s1, beta_s2, beta_d1, beta_d2, eps_ds, l0 = 80, if_bin = True):  #Return Dl_ds for two (or one) frequency
        l = np.arange(10000)
        Dl_d = A_d * (l/l0) ** alpha_d
        Dl_s = A_s * (l/l0) ** alpha_s
        f_s1,f_d1 = self.fg_spectra(nu1, beta_s1, beta_d1)
        f_s2,f_d2 = self.fg_spectra(nu2, beta_s2, beta_d2)
        Dl_cross = eps_ds *(f_d1*f_s2*np.sqrt(Dl_d * Dl_s) + f_d2*f_s1*np.sqrt(Dl_d * Dl_s)) 
        if if_bin == 1:
            lbin, output = get_binned(nside,bin,Dl_cross[:lmax+1],lmax)
        else:
            lbin,output = l,Dl_cross
        return  lbin,output
    

def get_fg_cross(nu1, nu2, lmax, bin, A_d, A_s, alpha_s, alpha_d, beta_s1, beta_s2, beta_d1, beta_d2, eps_ds, l0 = 80, if_bin = False):      #Eq(2.9)
    fg = fg_model()
    lbin, Dl_ds = fg.Dl_fg_cross_nu(nu1, nu2, lmax, bin, A_d, A_s, alpha_s, alpha_d, beta_s1, beta_s2, beta_d1, beta_d2, eps_ds, l0 = 80, if_bin = if_bin)
    return lbin, Dl_ds


def get_fg_total(nu1, nu2, lmax, bin, A_d, A_s, alpha_s, alpha_d, beta_s, beta_d, eps_ds, l0 = 80, if_bin = False):
    fg = fg_model()
    lbin, Dl_dd, Dl_ss = fg.Dl_fg_auto_nu(nu1, nu2,lmax, bin, A_d, A_s, alpha_s, alpha_d, beta_s, beta_d, l0 = l0, if_bin = if_bin)
    _,dl_ds = get_fg_cross(nu1, nu2, lmax, bin, A_d, A_s, alpha_s, alpha_d, beta_s, beta_s, beta_d, beta_d, eps_ds, l0 = l0, if_bin = if_bin)
    return lbin, Dl_dd + Dl_ss + dl_ds


def map_rescale(nside,map,trans,lmax=None,):
    if lmax == None:
        lmax=3*nside-1
    alm = hp.map2alm(map,lmax,lmax)
    alm_rescale = hp.almxfl(alm,cli(trans))
    map_rescale = hp.alm2map(alm_rescale,nside,lmax)
    return map_rescale


###################################################  Start from here  ########################################################

#########################################Prepare for the data and theory Cl###################################################
dl_tens_1, dl_lens_1= get_cl(1)
dl_tens_1_continue = dl_tens_1[:1000,2]    ############ tensor

l,dl_tens_1 = get_binned(nside,20,dl_tens_1_continue,999)
l,dl_lens_1 = get_binned(nside,20,dl_lens_1[:1000,2] ,999)

dl_tens_0, dl_lens_0= get_cl(0)
#dl_lens = dl_lens_0[:1000,2]     ############# lensing
l,dl_lens_0 = get_binned(nside,20,dl_lens_0[:1000,2] ,999)

#dl_lens = np.loadtxt('./data/lensedb.txt')

##########################################      Read data       ###################################################
data_vec = np.loadtxt('./data_vec_fit_average.txt') #[28,49]
COV_sim = np.load('./COV_matrix.npy')           #[49,28,28]

dl_lens = np.loadtxt('./LT_LT_mean.txt')[5]
with open('./dlbb_noise_cheat.pkl', 'rb') as f:   #注意：用cheat时需要改th_vec函数[0]
    dl_noise = pickle.load(f)

#with open('./dlbb_FG_average.pkl', 'rb') as f:
    #dl_FGs = pickle.load(f)

"""
vec = {(27,27):data_vec[0], (27,39):data_vec[1], (27,93):data_vec[2], (27,145):data_vec[3], (27,225):data_vec[4], (27,280):data_vec[5], (27,999):data_vec[6],
            (39,39):data_vec[7], (39,93):data_vec[8], (39,145):data_vec[9], (39,225):data_vec[10], (39,280):data_vec[11], (39,999):data_vec[12],
            (93,93):data_vec[13], (93,145):data_vec[14], (93,225):data_vec[15], (93,280):data_vec[16], (93,999):data_vec[17],
            (145,145):data_vec[18], (145,225):data_vec[19], (145,280):data_vec[20], (145,999):data_vec[21],
            (225,225):data_vec[22], (225,280):data_vec[23], (225,999):data_vec[24],
            (280,280):data_vec[25], (280,999):data_vec[26],
            (999,999):data_vec[27]}     #notice: Here "999" represents the LT for short

######################################### Cov matrix theroy ########################################################
COV_th = np.zeros((49, 28, 28))
for l in range(49):
    lbin = 10*(2*l+1)
    for i, key1 in enumerate(vec.keys()):
        for j, key2 in enumerate(vec.keys()):
            key3 = (key1[0],key2[0]) if key1[0] <= key2[0] else (key2[0],key1[0])
            key4 = (key1[1],key2[1]) if key1[1] <= key2[1] else (key2[1],key1[1])
            key5 = (key1[0],key2[1]) if key1[0] <= key2[1] else (key2[1],key1[0])
            key6 = (key1[1],key2[0]) if key1[1] <= key2[0] else (key2[0],key1[1])
            print(key1,key2,key3,key4,key5,key6)
            COV_th[l,i,j] = 1/(2*lbin+1) * (vec[key3][l] * vec[key4][l] + vec[key5][l] * vec[key6][l])
"""

fg = fg_model()

def get_fg_cross(nu1, nu2, lmax, bin, A_d, A_s, alpha_s, alpha_d, beta_s1, beta_s2, beta_d1, beta_d2, eps_ds, l0 = 80, if_bin = False):      #Eq(2.9)
    fg = fg_model()
    lbin, Dl_ds = fg.Dl_fg_cross_nu(nu1, nu2, lmax, bin, A_d, A_s, alpha_s, alpha_d, beta_s1, beta_s2, beta_d1, beta_d2, eps_ds, l0 = 80, if_bin = if_bin)
    return lbin, Dl_ds

def get_fg_total(nu1, nu2, lmax, bin, A_d, A_s, alpha_s, alpha_d, beta_s, beta_d, eps_ds, l0 = 80, if_bin = False):
    fg = fg_model()
    lbin, Dl_dd, Dl_ss = fg.Dl_fg_auto_nu(nu1, nu2,lmax, bin, A_d, A_s, alpha_s, alpha_d, beta_s, beta_d, l0 = l0, if_bin = if_bin)
    _,dl_ds = get_fg_cross(nu1, nu2, lmax, bin, A_d, A_s, alpha_s, alpha_d, beta_s, beta_s, beta_d, beta_d, eps_ds, l0 = l0, if_bin = if_bin)
    return lbin, Dl_dd + Dl_ss + dl_ds

nus = {27: 27 * 1e9, 39: 39 * 1e9, 93: 93 * 1e9, 145: 145 * 1e9, 225: 225 * 1e9, 280: 280 * 1e9}

def get_FG(nus,lmax, bin, A_d, A_s, alpha_s, alpha_d, beta_s, beta_d, eps_ds, if_bin = True):
    dl_FG = {}
    for i,nu in nus.items():
        for j,nu in nus.items():
            if i > j :
                continue
            key = (i,j)
            _,dl_FG[key] = get_fg_total(nus[i], nus[j], lmax, bin, A_d, A_s, alpha_s, alpha_d, beta_s, beta_d, eps_ds, if_bin = True)
    return dl_FG

def get_ds_bias(nus,lmax, bin, A_d, A_s, alpha_s, alpha_d, beta_s, beta_d, if_bin = True):
    dl_ds_bias = {}
    for i,nu in nus.items():
        for j,nu in nus.items():
            if i > j :
                continue
            key = (i,j)
            _,dl_ds_bias[key] = get_fg_cross(nus[i], nus[j], lmax, bin, A_d, A_s, alpha_s, alpha_d, beta_s, beta_s, beta_d, beta_d, 1, if_bin = True)
    return dl_ds_bias

###################################### Maximum likelihhod parameters ##########################################
A_d0 = 14.30 
alpha_d0 = -0.65
beta_d0 = 1.48

A_s0 = 2.40 
alpha_s0 = -0.80
beta_s0 = -3.10

eps_ds0 = 0

dl_FG_th = get_FG(nus, 1000, 20, A_d0, A_s0, alpha_s0, alpha_d0, beta_s0, beta_d0, eps_ds0, if_bin = True)
dl_cross_de_bias = get_ds_bias(nus, 1000, 20, A_d0, A_s0, alpha_s0, alpha_d0, beta_s0, beta_d0, if_bin = True)
"""
#####################################  Seperate noise from data sim ###########################################
dl_noise_cheat = {}
dl_noise_cheat[(27,27)] = data_vec[0] - dl_FG_th[(27,27)] - dl_lens 
dl_noise_cheat[(39,39)] = data_vec[7] - dl_FG_th[(39,39)] - dl_lens
dl_noise_cheat[(93,93)] = data_vec[13] - dl_FG_th[(93,93)] - dl_lens
dl_noise_cheat[(145,145)] = data_vec[18] - dl_FG_th[(145,145)] - dl_lens
dl_noise_cheat[(225,225)] = data_vec[22] - dl_FG_th[(225,225)] - dl_lens
dl_noise_cheat[(280,280)] = data_vec[25] - dl_FG_th[(280,280)] - dl_lens

dl_noise_cheat[(27,39)] = data_vec[1] - dl_FG_th[(27,39)] - dl_cross_de_bias[(27,39)] - dl_lens
dl_noise_cheat[(27,93)] = data_vec[2] - dl_FG_th[(27,93)] - dl_cross_de_bias[(27,93)] - dl_lens
dl_noise_cheat[(27,145)] = data_vec[3] - dl_FG_th[(27,145)] - dl_cross_de_bias[(27,145)] - dl_lens
dl_noise_cheat[(27,225)] = data_vec[4] - dl_FG_th[(27,225)] - dl_cross_de_bias[(27,225)] - dl_lens
dl_noise_cheat[(27,280)] = data_vec[5] - dl_FG_th[(27,280)] - dl_cross_de_bias[(27,280)] - dl_lens

dl_noise_cheat[(39,93)] = data_vec[8] - dl_FG_th[(39,93)] - dl_cross_de_bias[(39,93)] - dl_lens
dl_noise_cheat[(39,145)] = data_vec[9] - dl_FG_th[(39,145)] - dl_cross_de_bias[(39,145)] - dl_lens
dl_noise_cheat[(39,225)] = data_vec[10] - dl_FG_th[(39,225)] - dl_cross_de_bias[(39,225)] - dl_lens
dl_noise_cheat[(39,280)] = data_vec[11] - dl_FG_th[(39,280)] - dl_cross_de_bias[(39,280)] - dl_lens

dl_noise_cheat[(93,145)] = data_vec[14] - dl_FG_th[(93,145)] - dl_cross_de_bias[(93,145)] - dl_lens
dl_noise_cheat[(93,225)] = data_vec[15] - dl_FG_th[(93,225)] - dl_cross_de_bias[(93,225)] - dl_lens
dl_noise_cheat[(93,280)] = data_vec[16] - dl_FG_th[(93,280)] - dl_cross_de_bias[(93,280)] - dl_lens

dl_noise_cheat[(145,225)] = data_vec[19] - dl_FG_th[(145,225)] - dl_cross_de_bias[(145,225)] - dl_lens
dl_noise_cheat[(145,280)] = data_vec[20] - dl_FG_th[(145,280)] - dl_cross_de_bias[(145,280)] - dl_lens

dl_noise_cheat[(225,280)] = data_vec[23] - dl_FG_th[(225,280)] - dl_cross_de_bias[(225,280)] - dl_lens
"""

def get_th_vec(r, AL, noise, nus, bin, A_d, A_s, alpha_s, alpha_d, beta_s, beta_d, eps_ds, dl_bias=None, lmin=0, lmax=100):
    dl_tens_r = r*dl_tens_1    ############ tensor

    if dl_bias is None:
        dl_bias = 0*dl_tens_r

    dl_FG_th = get_FG(nus, 1000, 20, A_d, A_s, alpha_s, alpha_d, beta_s, beta_d, eps_ds, if_bin = True)
    dl_cross_de_bias = get_ds_bias(nus, 1000, 20, A_d, A_s, alpha_s, alpha_d, beta_s, beta_d, if_bin = True)

    dl_obs = {}
    dl_LT_cross = {}
    dl_LT_LT = AL * dl_lens
    #dl_FG = get_FG(nus,lmax, bin, A_d, A_s, alpha_s, alpha_d, beta_s, beta_d, eps_ds, if_bin = True)
    for i,_ in nus.items():
        for j,_ in nus.items():
            if i > j :
                continue
            key = (i,j)
            if key[0] != key[1]:
                dl_obs[key] = dl_tens_r + AL * dl_lens + noise[key] + dl_FG_th[key] #+ dl_cross_de_bias[key]     #################注意：这次fg模拟发现没有cross bias项，待检查
            else :
                dl_obs[key] = dl_tens_r + AL * dl_lens + noise[key] + dl_FG_th[key]
    for i,_ in nus.items():
            dl_LT_cross[i] = np.sqrt(AL) * dl_lens 
    
    th_vec = np.array([dl_obs[(27,27)],dl_obs[(27,39)],dl_obs[(27,93)],dl_obs[(27,145)],dl_obs[(27,225)],dl_obs[(27,280)],dl_LT_cross[27],
            dl_obs[(39,39)],dl_obs[(39,93)],dl_obs[(39,145)],dl_obs[(39,225)],dl_obs[(39,280)],dl_LT_cross[39],
            dl_obs[(93,93)],dl_obs[(93,145)],dl_obs[(93,225)],dl_obs[(93,280)],dl_LT_cross[93],
            dl_obs[(145,145)],dl_obs[(145,225)],dl_obs[(145,280)],dl_LT_cross[145],
            dl_obs[(225,225)],dl_obs[(225,280)],dl_LT_cross[225],
            dl_obs[(280,280)],dl_LT_cross[280],
            dl_LT_LT])
    
    return th_vec
    

#Theory vector
th_vec = get_th_vec(0, 1, dl_noise, nus, 20, A_d0, A_s0, alpha_s0, alpha_d0, beta_s0, beta_d0, eps_ds0, lmin=0, lmax=100)



##############################################  Define likelihood ###################################################
COV_INV = np.linalg.inv(COV_sim)
COV_DET = np.log(np.linalg.det(COV_sim))


def gauss_cl_logp_th_cov(r, AL, A_d, A_s, alpha_d,  alpha_s,  beta_s, beta_d, dl_noise=dl_noise, nus=nus, eps_ds=0, bin=20, dl_bias=None, lmin=0, lbin_max = 10):
    """
    Defines a gaussian likelihood in log.
    """
    likelihood = 0
    X_theory = get_th_vec(r, AL, dl_noise, nus, 20, A_d, A_s, alpha_s, alpha_d, beta_s, beta_d, eps_ds, dl_bias=None, lmin=0, lmax=200)
    for i in np.arange(lbin_max):
        likelihood_l = np.dot(np.dot((th_vec[:,i] - X_theory[:,i]).T, COV_INV[i,:,:] ), (th_vec[:,i] - X_theory[:,i])) + COV_DET[i]
        #likelihood_l = np.dot(np.dot((X_data[:,i] - X_theory[:,i]).T, cli(cal_cov_3(X_data,l[i],fsky=0.8,)[:,:,i])), (X_data[:,i] - X_theory[:,i])) 
        likelihood += -0.5*likelihood_l
    return likelihood

def gauss_cl_logp_sim_cov(r,AL,  A_d, A_s, alpha_d, alpha_s,  beta_s, beta_d, dl_noise=dl_noise, nus=nus, eps_ds=0, bin=20, dl_bias=None, lmin=0, lbin_max = 10):
    """
    Defines a gaussian likelihood in log.
    """
    likelihood = 0
    X_theory = get_th_vec(r, AL, dl_noise, nus, 20, A_d, A_s, alpha_s, alpha_d, beta_s, beta_d, eps_ds, dl_bias=None, lmin=0, lmax=200)
    for i in np.arange(1,lbin_max):
        likelihood_l = np.dot(np.dot((data_vec[:,i] - X_theory[:,i]).T, COV_INV[i,:,:] ), (data_vec[:,i] - X_theory[:,i])) + COV_DET[i]
        #likelihood_l = np.dot(np.dot((X_data[:,i] - X_theory[:,i]).T, cli(cal_cov_3(X_data,l[i],fsky=0.8,)[:,:,i])), (X_data[:,i] - X_theory[:,i])) 
        likelihood += -0.5*likelihood_l
    return likelihood


######################################################################################################################
#################################################  Here we go ########################################################



import warnings
from mpi4py import MPI
from cobaya import run
from cobaya.log import LoggedError

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Filter out RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Define the info dictionary
info = {"likelihood": {"r&AL": gauss_cl_logp_sim_cov}}

info["params"] = {
    "r": {"prior": {"min": -0.1, "max": 0.1}, "ref": 0, "proposal": 0.001, "latex": r"r"},
    "AL": {"prior": {"min": 0.8, "max": 1.2}, "ref": 1, "proposal": 0.001, "latex": r"A_L"},
    "A_d": {"prior": {"min": 10, "max": 16}, "ref": 14.30, "proposal": 0.05, "latex": r"A_d"},
    "alpha_d": {"prior": {"min": -0.9, "max": -0.3}, "ref": -0.65, "proposal": 0.005, "latex": r"\alpha_d"},
    "beta_d": {"prior": {"min": 1.3, "max": 1.8}, "ref": 1.48, "proposal": 0.001, "latex": r"\beta_d"},
    "A_s": {"prior": {"min": 1, "max": 4}, "ref": 2.40, "proposal": 0.005, "latex": r"A_s"},
    "alpha_s": {"prior": {"min": -1.4, "max": 0}, "ref": -0.80, "proposal": 0.003, "latex": r"\alpha_s"},
    "beta_s": {"prior": {"min": -3.3, "max": -2.7}, "ref": -3.1, "proposal": 0.001, "latex": r"\beta_s"},
}

# Add the MCMC sampler information
info["sampler"] = {
    "mcmc": {
        "Rminus1_stop": 1e-3,
        "max_tries": 1000000000,
    }
}

# Run the MCMC sampling
success = False
try:
    upd_info, mcmc = run(info)
    success = True
except LoggedError as err:
    pass

# Check if the sampling was successful
success = all(comm.allgather(success))

if not success and rank == 0:
    print("Sampling failed!")


# Run the MCMC sampling again with the updated info
#updated_info, sampler = run(info)

# Gather and combine the chains from all MPI processes, skipping the first third of each chain
full_chain = mcmc.samples(combined=True, skip_samples=0.33, to_getdist=True)

# 保存 full_chain2
with open('full_chain_lens.pkl', 'wb') as f:
    pickle.dump(full_chain, f)

# Analyze and plot
mean = full_chain.getMeans()[:8]
covmat = full_chain.getCovMat().matrix[:8, :8]
print("Mean:")
print(mean)
print("Standard error:")
print(np.sqrt(covmat))

# %matplotlib inline  # Uncomment if running from a Jupyter notebook
import getdist.plots as gdplt

gdplot = gdplt.get_subplot_plotter()
gdplot.triangle_plot(full_chain, ["r", "AL", "A_d", "alpha_d", "beta_d", "A_s", "alpha_s", "beta_s"], filled=True, contour_colors=['red'])

for i in range(8):
    gdplot.subplots[i, i].axvline(x=mean[i], color='r', linestyle=':')
    gdplot.subplots[i, i].text(mean[i], 1, f"Mean: {mean[i]:.6f}", fontsize=6, color='green', ha='left', va='top')
    gdplot.subplots[i, i].text(mean[i], 0.9, f"Sigma: {np.sqrt(covmat[i])[i]:.6f}", fontsize=6, color='green', ha='left', va='top')

gdplot.add_text("lens_sim, cov_sim, ", x=0, y=3)
gdplot.export('gaussian0_cheat_20_200.png', dpi=300)

