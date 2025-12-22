import healpy as hp
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy import units as u
from astropy.cosmology import Planck15
from scipy.interpolate import griddata

#from spectral import *
#from ilc_class import *

####################################    CIB    ######################################

def cib_spectral_response(freqs, Tdust_CIB=8.35, beta_CIB=1.59): #input frequency in GHz
    # function from pyilc
    # CIB = modified blackbody here
    # N.B. overall amplitude is not meaningful here; output ILC map (if you tried to preserve this component) would not be in sensible units

    TCMB = 2.726 #Kelvin
    TCMB_uK = 2.726e6 #micro-Kelvin
    hplanck=6.626068e-34 #MKS
    kboltz=1.3806503e-23 #MKS
    clight=299792458.0 #MKS

    # function needed for Planck bandpass integration/conversion following approach in Sec. 3.2 of https://arxiv.org/pdf/1303.5070.pdf
    # blackbody derivative
    # units are 1e-26 Jy/sr/uK_CMB
    def dBnudT(nu_ghz):
        nu = 1.e9*np.asarray(nu_ghz)
        X = hplanck*nu/(kboltz*TCMB)
        return (2.*hplanck*nu**3.)/clight**2. * (np.exp(X))/(np.exp(X)-1.)**2. * X/TCMB_uK

    # conversion from specific intensity to Delta T units (i.e., 1/dBdT|T_CMB)
    #   i.e., from W/m^2/Hz/sr (1e-26 Jy/sr) --> uK_CMB
    #   i.e., you would multiply a map in 1e-26 Jy/sr by this factor to get an output map in uK_CMB
    def ItoDeltaT(nu_ghz):
        return 1./dBnudT(nu_ghz)


    nu0_CIB_ghz = 353.0    #CIB pivot frequency [GHz]
    kT_e_keV = 5.0         #electron temperature for relativistic SZ evaluation [keV] (for reference, 5 keV is rough temperature for a 3x10^14 Msun/h cluster at z=0 (e.g., Arnaud+2005))
    nu0_radio_ghz = 150.0  #radio pivot frequency [GHz]
    beta_radio = -0.5      #radio power-law index

    nu_ghz = freqs
    nu = 1.e9*np.asarray(nu_ghz).astype(float)
    X_CIB = hplanck*nu/(kboltz*Tdust_CIB)
    nu0_CIB = nu0_CIB_ghz*1.e9
    X0_CIB = hplanck*nu0_CIB/(kboltz*Tdust_CIB)
    resp = (nu/nu0_CIB)**(3.0+(beta_CIB)) * ((np.exp(X0_CIB) - 1.0) / (np.exp(X_CIB) - 1.0)) * (ItoDeltaT(np.asarray(nu_ghz).astype(float))/ItoDeltaT(nu0_CIB_ghz))
    #resp = (nu)**(3.0+(beta_CIB)) * (1 / (np.exp(X_CIB) - 1.0)) * (ItoDeltaT(np.asarray(nu_ghz).astype(float)))
    #resp[np.where(nu_ghz == None)] = 0. #this case is appropriate for HI or other maps that contain no CMB-relevant signals (and also no CIB); they're assumed to be denoted by None in nu_ghz
    return resp



def cib_spectra_relative(freqs, Tdust_CIB, beta_CIB, A):
    cl = cib_spectral_response(freqs, Tdust_CIB, beta_CIB)**2 * A    # Notice: A is the amplitude of the CIB component, independent of frequency, but depends on ell
    return cl


def CIB_interpolate(freq0, freq_out, map0, Tdust_CIB=8.35, beta_CIB=1.59):
    resp0 = cib_spectral_response(freq0, Tdust_CIB, beta_CIB)
    resp_out = cib_spectral_response(freq_out, Tdust_CIB, beta_CIB)
    print(f'CIB scale factor = {resp_out / resp0} ({freq0}GHz ->{freq_out}GHz)')
    return map0 * resp_out / resp0
#Tdust_CIB_fit = 8.35                                       ##### This is fitted from the mmDL maps at six frequencies, see fit_and_interp.ipynb !!!!
#beta_CIB_fit = 1.59
#map30_inter = CIB_interpolate(148, 280, map_ir_148, Tdust_CIB_fit, beta_CIB_fit)


####################################    RADIO PS    ######################################

def radio_spectral_response(freqs, beta_g): #input frequency in GHz
    T_cmb = 2.726
    h = 6.62607004*10**(-34)
    kb = 1.38064852*10**(-23)
    f = 1. #fsky
    response = []
    for freq in freqs:
        nu = freq*10**9 * h / kb / T_cmb   # notice: the constant is irrelevant here, just for the convience of curve fitting
        response.append((nu**beta_g))
    return np.array(response)

def radio_spectral_response_poly(freqs, a1=-2.7714227059838192, a2=0.019242069853106277, a3=0.36813523402646997): #input frequency in GHz
    T_cmb = 2.726
    h = 6.62607004*10**(-34)
    kb = 1.38064852*10**(-23)
    f = 1. #fsky
    response = []
    for freq in freqs:
        nu = freq*10**9  * h / kb / T_cmb   # notice: the constant is irrelevant here, just for the convience of curve fitting
        response.append(np.exp(0 + a1*np.log(nu) + a2*np.log(nu)**2 + a3*np.log(nu)**3))  # set a0 = 0
    return np.array(response)

def radio_spectra_relative(freqs, beta_g, A):
    cl = radio_spectral_response(freqs, beta_g)**2 * A
    return cl

def radio_spectra_relative_poly(freqs,  a1, a2, a3, A):
    cl = radio_spectral_response_poly(freqs,  a1, a2, a3)**2 * A
    return cl


def radio_spectral_interpolate_poly(freq0, freq_out, map0, a1=-2.7714227059838192, a2=0.019242069853106277, a3=0.36813523402646997): #input frequency in GHz
    resp0 = radio_spectral_response_poly([freq0], a1, a2, a3)
    resp_out = radio_spectral_response_poly([freq_out], a1, a2, a3)
    print(f'Radio scale factor = {resp_out / resp0} ({freq0}GHz ->{freq_out}GHz)')
    return map0 * resp_out / resp0    
#a1, a2, a3 = (-2.7714227059838192, 0.019242069853106277, 0.36813523402646997)     ##### This is fitted from the mmDL maps at six frequencies, see fit_and_interp.ipynb !!!!
#map30_inter = radio_spectral_interpolate_poly([148], [27], map_radio_148, a1, a2, a3)



####################################    tSZ    ######################################

def tsz_spectral_response(freqs): #input frequency in GHz
    T_cmb = 2.726
    h = 6.62607004*10**(-34)
    kb = 1.38064852*10**(-23)
    f = 1. #fsky
    response = []
    for freq in freqs:
        x = h*(freq*10**9)/(kb*T_cmb) #x is v/56.9 GHz
        response.append(T_cmb*(x*1/np.tanh(x/2)-4)) #was factor of tcmb microkelvin before
    return np.array(response)

def tSZ_interpolate(freq0, freq_out, map0):
    resp0 = tsz_spectral_response([freq0])[0]
    resp_out = tsz_spectral_response([freq_out])[0]
    print(f'tSZ scale factor = {resp_out / resp0} ({freq0}GHz ->{freq_out}GHz)')
    return map0 * resp_out / resp0
#map30_inter = tSZ_interpolate([148], [27], map_radio_148,)