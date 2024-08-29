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

from utils_mine import *

from lenspyx.utils import camb_clfile
from lenspyx.utils_hp import synalm, almxfl, alm2cl

camb_path = os.path.realpath(os.path.join(os.getcwd(),'..'))    
sys.path.insert(0,camb_path)

import camb
from camb import model, initialpower ,correlations


from scipy.special import factorial

def making_maps(nside, lmax=None,dlmax=None,nrms_f=None,fwhm_f=None,phi_map=None,pixwin=True,seed=0):
    print('setting parameters...')

    np.random.seed(seed)  

    if lmax == None:
        lmax = 3*nside-1
    lmax_len = lmax # desired lmax of the lensed field.
    if dlmax == None:
        dlmax = lmax   # lmax of the unlensed fields is lmax + dlmax.  (some buffer is required for accurate lensing at lmax)        
    epsilon = 1e-6  # target accuracy of the output maps (execution time has a fairly weak dependence on this)

    if nside == 1024:
        lmax_unl, mmax_unl = lmax_len + dlmax  , lmax_len + dlmax 
    else:
        lmax_unl, mmax_unl = 7000  , 7000
    Tcmb = 2.7255e6

    cl = camb_clfile('/sharefs/alicpt/users/chenwz/download/plancklens/plancklens/data/cls/FFP10_wdipole_lenspotentialCls.dat',lmax_unl)   #l,TT,EE,BB,TE,PP,TP,EP
    if phi_map is not None:
        pmap = phi_map  
        plm = hp.map2alm(pmap, lmax=lmax + dlmax, verbose=False)
    else:
        plm = hp.synalm(cl['pp'], lmax=lmax + dlmax, new=True, verbose=False)
        pmap = hp.alm2map(plm, nside, verbose=False)

    #unlensed CMB 球谐系数
    tlm_unl, elm_unl, blm_unl = hp.synalm(  [cl['tt'], cl['ee'], cl['bb'], cl['te']], lmax=lmax + dlmax, new=True, verbose=False)

    print('lensing...')
    #potential 球谐系数
    #klm = hp.map2alm(Kmap, lmax=2000)
    #q2k = lambda l: l*(l + 1) / 2 # potential -> convergence
    #plm = hp.almxfl(klm, cli(q2k(np.arange(2000 + 1, dtype=float))))   #lensing potential

    # We then transform the lensing potential into spin-1 deflection field, (见下面 deflect the temperature map.)
    dlm = hp.almxfl(plm, np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2)))

    # Geometry on which to produce the lensed map
    geom_info = ('healpix', {'nside':nside}) # here we will use an Healpix grid with nside 2048
    Tlen = lenspyx.lensing.alm2lenmap(tlm_unl, dlm,  geometry=geom_info, verbose=False)#alm2lenmap：alm_unl * dlm -> lensed map
    Qlen, Ulen = lenspyx.lensing.alm2lenmap_spin(elm_unl, dlm, 2, geometry=geom_info, verbose=False)
    #Tlen, Qlen, Ulen = lenspyx.alm2lenmap([tlm_unl, elm_unl, blm_unl], dlm, geometry=geom_info, verbose=1, epsilon=epsilon)
    if fwhm_f is not None:
        '''
        Tlen = hp.smoothing(Tlen, fwhm=fwhm)
        Qlen = hp.smoothing(Qlen, fwhm=fwhm)
        Ulen = hp.smoothing(Ulen, fwhm=fwhm)
        '''
        # transfer function NOTE pixwin func is included as well
        if len(fwhm_f) == 0 or 1:
            fwhm_f = fwhm_f[0] if fwhm_f else 0.
            transf = bl(fwhm_f, nside=nside, lmax=lmax, pixwin=True)
        else:
            assert nrms_f
            transf = bl_eft(nrms_f, fwhm_f, nside=nside, lmax=lmax, pixwin=True)  #Effective beam function, used when combine more than one experiments
        tlm_len = hp.map2alm(Tlen, lmax=lmax)
        elm_len, blm_len = hp.map2alm_spin([Qlen, Ulen], 2, lmax=lmax)
    
        # Convolution with transfer function    #tranf = H_l*B_l, 见PLANCK2015 (A.4)，亦见PLANCK2013 (38)式
        Tlen = hp.alm2map(hp.almxfl(tlm_len, transf, inplace=True), nside, verbose=False)
        Qlen, Ulen = hp.alm2map_spin([hp.almxfl(elm_len, transf), hp.almxfl(blm_len, transf)], nside, 2, lmax)

    return Tlen, Qlen, Ulen, pmap


def gaussian_cib(nside,phi_map,rho,clii):
    lmax = 3*nside-1
    plm = hp.map2alm(phi_map,lmax,)
    clpp = hp.alm2cl(plm,plm)
    clip = rho * np.sqrt(clpp*clii)
    ilm = hp.almxfl(plm,clip*cli(clpp))
    clnn = clii - (clip**2)*cli(clpp)
    nlm =  synalm(clnn, lmax=lmax, mmax=lmax)
    Imap = hp.alm2map(ilm + nlm, nside,)
    noise_map = hp.alm2map(nlm, nside,)
    return Imap,noise_map


def making_maps_camb(nside,r,fwhm=None):

    print('setting parameters...')
    nside = nside
    lmax = 3*nside-1
    lmax_len = 3*nside-1 # desired lmax of the lensed field.
    dlmax = 3*nside-1   # lmax of the unlensed fields is lmax + dlmax.  (some buffer is required for accurate lensing at lmax)
    epsilon = 1e-6  # target accuracy of the output maps (execution time has a fairly weak dependence on this)
    lmax_unl, mmax_unl = lmax_len + dlmax  , lmax_len + dlmax 
    if nside == 2048 or nside == 4096:
        lmax_unl, mmax_unl = 10000  , 10000
    Tcmb = 2.7255e6

    #temporary used for generating the input Cls
    #运行Camb，给出理论计算结果
    print('running CAMB...')
    #Set up a new set of parameters for CAMB
    #对象 pars 用于存储和设置 CAMB 的参数
    pars = camb.CAMBparams()
    #This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency

    #parameters from planck FFP10_wdipole_params.ini
    #通过调用 set_cosmology 方法设置宇宙学参数。其中： H0为Hubble 常数、ombh2为重子物质密度参数、omch2为冷暗物质密度参数、mnu为中微子的质量、omk为曲率参数 、tau为再电离光深度 
    pars = camb.set_params(H0=67.01904, ombh2=0.02216571, omch2=0.1202944, mnu=0.0006451439, omk=0, tau=0.06018107, YHe=0.2453006, w=-1.0, wa=0, nnu=3.046, nt=-r/8.0, nrun=0,\
                       pivot_scalar=0.05, pivot_tensor=0.05, )

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

    unlensed_total = results.get_unlensed_total_cls(raw_cl=True) * Tcmb**2 # unlensed CMB power spectra, including tensors

    dl_ppte=results.get_lens_potential_cls(lmax=None, CMB_unit='muK', raw_cl=False)   #numpy array CL[0:lmax+1,0:3], where 0..2 indexes PP, PT, PE.(T,E均unlensed)
    dlpp=dl_ppte[:,0]
    lensed_cl = results.get_lensed_cls_with_spectrum(dlpp, lmax=None, CMB_unit=None, raw_cl=True) * Tcmb**2

    #unlensed CMB 球谐系数
    tlm_unl = synalm(unlensed_total[:,0], lmax=lmax_unl, mmax=mmax_unl)
    elm_unl = synalm(unlensed_total[:,1], lmax=lmax_unl, mmax=mmax_unl)
    blm_unl = synalm(unlensed_total[:,2], lmax=lmax_unl, mmax=mmax_unl)

    plm = hp.synalm(clpp,lmax=dlmax,new=True)
    pmap = hp.alm2map(plm, nside, lmax=dlmax, verbose=False)

    # We then transform the lensing potential into spin-1 deflection field, (见下面 deflect the temperature map.)
    dlm = almxfl(plm, np.sqrt(np.arange(dlmax + 1, dtype=float) * np.arange(1, dlmax + 2)), None, False)  

    # Geometry on which to produce the lensed map
    geom_info = ('healpix', {'nside':nside}) # here we will use an Healpix grid with nside 2048
    Tlen, Qlen, Ulen = lenspyx.alm2lenmap([tlm_unl, elm_unl, blm_unl], dlm, geometry=geom_info, verbose=1, epsilon=epsilon)

    if fwhm is not None:
        Tlen = hp.smoothing(Tlen, fwhm=fwhm)
        Qlen = hp.smoothing(Qlen, fwhm=fwhm)
        Ulen = hp.smoothing(Ulen, fwhm=fwhm)

    
    return Tlen, Qlen, Ulen, pmap, lensed_cl, clpp