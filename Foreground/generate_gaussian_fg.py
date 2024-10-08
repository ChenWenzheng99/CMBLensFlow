import os,sys
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

sys.path.append("/sharefs/alicpt/users/chenwz/Testarea/FORECAST/Foreground/run_with_foreground")
from utils_mine import *

import pysm3 as pysm
import pysm3.units as u
import pymaster as nmt

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
        A factor concerning frequency dependence of the fg spectra.
        """
        h = 6.62607015e-34
        k = 1.380649e-23
        Tcmb = 2.7255
        Td = 19.6   #in K
        f_s = (nu/nu0_s) ** beta_s      / (self.unit_convert(nu)/self.unit_convert(nu0_s))                            #If you use uKCMB unit, you should add this unit conversion.
        f_d = (nu/nu0_d) ** (beta_d + 1) * (np.exp(h*nu0_d/k/Td)-1) / (np.exp(h*nu/k/Td)-1)     / (self.unit_convert(nu)/self.unit_convert(nu0_d))
        return f_s, f_d
    
    def Dl_fg_nu(self, lmax, bin, A, alpha, l0 = 80, if_bin = True):  # We only consdier BB power spectrum here.
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

def decon_alm(alm,fwhm,lmax=None):
    """
    deconvolve alm with beam
    """
    bl = hp.gauss_beam(fwhm,lmax=lmax)
    alm_deconv = hp.almxfl(alm,cli(bl))
    return alm_deconv

def decon_map(nside,map,fwhm,lmax=None):
    """
    deconvolve map with beam
    """
    alm = hp.map2alm(map)
    alm_deconv = decon_alm(alm,fwhm,lmax)
    map_deconv = hp.alm2map(alm_deconv,nside)
    return map_deconv

def qu2eb_alm(nside,lmax,qlm,ulm):
    """convert Q/U alm to E/B alm, only correct for full sky, but leads negligible corruption to partial sky E mode"""
    Qmap = hp.alm2map(qlm,nside)
    Umap = hp.alm2map(ulm,nside)
    elm,blm = curvedsky.utils.hp_map2alm_spin(nside,lmax,lmax,2,Qmap,Umap,)
    return elm,blm

def qu2eb_map(nside,lmax,Qmap,Umap):
    """convert Q/U map to E/B map, only correct for full sky, but leads negligible corruption to partial sky E mode"""
    elm,blm = hp.map2alm_spin([Qmap,Umap],2)
    Emap = hp.alm2map(elm,nside)
    Bmap = hp.alm2map(blm,nside)
    return Emap,Bmap

def eb2qu_map(nside,lmax,Emap,Bmap):
    """convert E/B map to Q/U map, only correct for full sky, but leads negligible corruption to partial sky E mode"""
    elm = curvedsky.utils.hp_map2alm(nside,lmax,lmax,Emap,)
    blm = curvedsky.utils.hp_map2alm(nside,lmax,lmax,Bmap,)
    Q,U = curvedsky.utils.hp_alm2map_spin(nside, lmax, lmax, 2, elm, blm)
    return Q,U
    

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


def gaussian_fg(nside, nus, seed, same=False, method=1, A_d_b = 14.30, alpha_d_b = -0.65, A_s_b = 2.40, alpha_s_b = -0.80, beta_d = 1.48, beta_s = -3.10, eps_ds = 0 ):
    """
    nus: All the frequencies you want to use in a single simulation.

    # NOTICE
    1. The seed of all map in one simulation should be the same, otherwise you can't recover the fg power spectrum when crossing.
    2. For a single frequency, you can generate dust map and synchrotron map seperately, and combine them to the total fg map (Method1). 
    3. If you use method1, for nu1 map and nu2 map crossing, there will be an extra bias term (d1*s2 + d2*s1), which corresponds to an epsilon_ds=1 cross term, we should add this in our theory Cl when fitting.
        Notice: This bias term vanish when nu1 = nu2 , do not add this bias term to auto-power spectrum !!!!
    4. Alternatively, you can generate a gaussian realization from the total fg power spectrum (Method2).   In this way, you also need to modify the extra bias when fitting parameters.
    """

    nside = nside
    lmax = 3 * nside - 1

    ell = np.arange(lmax+1)
    cl_norm = ell*(ell+1)/np.pi/2
    zeros = np.zeros(len(ell), dtype=np.double)

    bin = 20

    #I  # We fit the 7-parameters for Ali sky fg intensity, finding that A_d_t0 = 60 and A_s_t0 = 9.7 with other paramters almost the same as the pol ones.
    A_d_t0 = 3*2**0.5*A_d_b  # This value suits for Ali sky
    alpha_d_t0 = alpha_d_b
    A_s_t0 = 3*2**0.5* A_s_b  # This value suits for Ali sky
    alpha_s_t0 = alpha_s_b

    #B mode # For Ali sky, the 7-params for fg pol are: A_d_b0 = 14.3, alpha_d_b0 = -0.65, beta_d0 = 1.48, A_s_b0 = 2.4, alpha_s_b0 = -0.8, beta_s0 = -3.1, eps_ds0 = 0.001(we fixed to 0 in our procedure).
    A_d_b0 = A_d_b
    alpha_d_b0 = alpha_d_b
    A_s_b0 = A_s_b
    alpha_s_b0 = alpha_s_b

    #E mode : Simply the same as the B-mode
    A_d_e0 = A_d_b 
    alpha_d_e0 = alpha_d_b
    A_s_e0 = A_s_b
    alpha_s_e0 = alpha_s_b

    beta_d0 = beta_d
    beta_s0 = beta_s

    eps_ds0 = eps_ds

    
    fg_nu = {}
    for i,nu in nus.items():
        if method == 1:
            #First, get the fg power spectrum
            _,Dl_dust_EE = get_fg_total(nu, nu, lmax, 1, A_d_e0, 0, 0, alpha_d_e0, 0, beta_d0, eps_ds0, l0 = 80, if_bin = False)
            _,Dl_dust_BB = get_fg_total(nu, nu, lmax, 1, A_d_b0, 0, 0, alpha_d_b0, 0, beta_d0, eps_ds0, l0 = 80, if_bin = False)
            Dl_dust_EE[0] = 0
            Dl_dust_BB[0] = 0

            _,Dl_synch_EE = get_fg_total(nu, nu, lmax, 1, 0, A_s_e0, alpha_s_e0, 0, beta_s0, 0, eps_ds0, l0 = 80, if_bin = False)
            _,Dl_synch_BB = get_fg_total(nu, nu, lmax, 1, 0, A_s_b0, alpha_s_b0, 0, beta_s0, 0, eps_ds0, l0 = 80, if_bin = False)
            Dl_synch_EE[0] = 0
            Dl_synch_BB[0] = 0
            Dl_total_BB = Dl_dust_BB + Dl_synch_BB
            #Second, generate the fg map with the same seed
            #dust
            np.random.seed(seed)
            m_sigma_G1 = hp.synfast([
                zeros[:lmax+1],
                Dl_dust_EE[:lmax+1]  / cl_norm[:lmax+1], Dl_dust_BB[:lmax+1]  / cl_norm[:lmax+1],
                zeros[:lmax+1], zeros[:lmax+1], zeros[:lmax+1]], 2048, new=True)   
            m_sigma_G1_T = hp.synfast(3*2**0.5*Dl_dust_EE[1:lmax+1] / cl_norm[1:lmax+1], nside, new=True)   #避免l=0的nan
            #hp.mollview(m_sigma_G1[2])

            #synch
            if same == True:
                np.random.seed(seed)
            else:
                np.random.seed(seed + 9999)
            m_sigma_G2 = hp.synfast([
                zeros[:lmax+1],
                Dl_synch_EE[:lmax+1]  / cl_norm[:lmax+1], Dl_synch_BB[:lmax+1]  / cl_norm[:lmax+1],
                zeros[:lmax+1], zeros[:lmax+1], zeros[:lmax+1]], 2048, new=True) 
            m_sigma_G2_T = hp.synfast(3*2**0.5*Dl_synch_EE[1:lmax+1]  / cl_norm[1:lmax+1], nside, new=True)   #避免l=0的nan
            #hp.mollview(m_sigma_G2[2]) 
            fg_nu[i] = [m_sigma_G1_T+m_sigma_G2_T, m_sigma_G1[1] + m_sigma_G2[1], m_sigma_G1[2] + m_sigma_G2[2]]
        
        else:
            # Or you can use the following code to generate fg map:
            # First, get the total fg power spectrum:
            _,Dl_total_EE = get_fg_total(nu, nu, lmax, 1, A_d_e0, A_s_e0, alpha_s_e0, alpha_d_e0, beta_s0, beta_d0, eps_ds0, l0 = 80, if_bin = False)
            _,Dl_total_BB = get_fg_total(nu, nu, lmax, 1, A_d_b0, A_s_b0, alpha_s_b0, alpha_d_b0,  beta_s0, beta_d0, eps_ds0, l0 = 80, if_bin = False)
            Dl_total_EE[0] = 0
            Dl_total_BB[0] = 0
            #Second, generate the fg map with the same seed
            np.random.seed(seed)
            m_sigma_G = hp.synfast([
                zeros[:lmax+1],
                Dl_total_EE[:lmax+1]  / cl_norm[:lmax+1], Dl_total_BB[:lmax+1]  / cl_norm[:lmax+1],
                zeros[:lmax+1], zeros[:lmax+1], zeros[:lmax+1]], 2048, new=True)   
            m_sigma_G_T = hp.synfast(3*2**0.5*Dl_total_EE[1:lmax+1] / cl_norm[1:lmax+1], nside, new=True)
            fg_nu[i] = [m_sigma_G_T, m_sigma_G[1], m_sigma_G[2]]
    return  fg_nu,Dl_total_BB
