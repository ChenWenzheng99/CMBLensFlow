#load modules
print('loading modules...')
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

##########################################################################

import os,sys
import pylab as pl
import numpy as np
import lenspyx
import healpy as hp

from lenspyx.utils import camb_clfile
from lenspyx.utils_hp import synalm, almxfl, alm2cl

camb_path = os.path.realpath(os.path.join(os.getcwd(),'..'))    
sys.path.insert(0,camb_path)

import camb
from camb import model, initialpower ,correlations

from scipy.special import factorial



def noise(nlev, nside,  seed=0, fwhm_f=None):   #The inhomogeneous instrumental noises is generated based on AliCPT noise variance map.
    """ Noise simulations, we include
            - white noise
            - noise realization according to given noise variance map
            - noise realization and combination of multi-channels

        * nlev      : it depends ...,输入T的,单位muK-pix
        * nside     : nside
        * savePath  : directory to save data
        * seed      : random seed
        * fwhm_f    : only for multi-channel combination

        All AliCPT's detectors are polarized and thus for simplicity
        nlev_Q = nlev_T * sqrt(2)
    """
    #assert isinstance(nlev, list or float or int)   #检查输入的nlev数据类型
    npix = hp.nside2npix(nside)
    fname = 'map_noise_nside%d_%04d.fits'

    np.random.seed(seed)
    m = np.random.normal(size=(3, npix)) * nlev \
                * np.array([1, 2 ** 0.5, 2 ** 0.5]).reshape(3,1)     #三行分别是T,Q,U的
    #variance map 里存的不是variance,而是standard deviation(标准差)，这里生成了符合noise标准差的高斯分布的随机数，得到variance map
    return m


def homo_noise(nlev, nside, seed=0, fwhm_f=None):

    nside = nside
    lmax = 3*nside-1
    npix = hp.nside2npix(nside)

    uKamin2uKpix = lambda n, npix : n / np.sqrt((360 * 60) ** 2 / np.pi / npix)  # 1° = pi/180 rad, 1' (arcmin) = pi/180/60 rad, 见PLANCK2015 (A.7)
    uKpix2uKamin = lambda n, npix : n * np.sqrt((360 * 60) ** 2 / np.pi / npix)
    noise_pix = uKamin2uKpix(nlev, npix)  #muK-pix
    noise_map = noise(noise_pix,nside,seed=seed)    #noise map
    return noise_map


def obs_noise(nlev, nside, seed=0, N_red_T=None, l_knee_T=None, alpha_knee_T=None, n_red_P=None, l_knee_P=None, alpha_knee_P=None, if_red=False, scale_factor = 2):
    """
    Generate 1/f noise and white noise, Gaussian realization from N_noise = N_red * (l / l_knee)**alpha_knee + N_white.

    !!!!!!!!!!!!!!          Notice:         !!!!!!!!!!!!!!!

    since the limited lmax of a healpy map (the limited resolution eliminate smaller modes, which should also exist in the pixel domain noise), 
    you are adviced to first generate a 2*nside map (or larger), then decrease to nside. 
    Otherwise, the generated noise map have a correct noise power, but will be with smaller sigma_pixel compared to theory value, a scale factor (theory / sim) is approximately as follows:

    Original nside      ->      Desired output nside        : 1.160
    nside               ->      nside                       : 1.160
    2*nside             ->      nside                       : 1.015  (recommended)
    4*nside             ->      nside                       : 1.009  (recommended)
    8*nside             ->      nside                       : 1.004

    We see at least a 2*nside map is needed to get a reasonable noise map.

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    

    
    * nlev : noise level of white noise in uK-arcmin.
    * N_red_T : 1/f noise power for temperature, in uK^2-s
    * l_knee_T : Specific scale where white noise becomes dominant.
    * alpha_knee_T : Power index control the decay of 1/f noise .
    * n_red_P : 1/f noise level for polarization, in uK-arcmin. If None, it will be set to nlev.
    * l_knee_P : Specific scale where white noise becomes dominant.
    * alpha_knee_P : Power index control the decay of 1/f noise .
    * if_red : If True, include 1/f noise, otherwise only white noise.
    * scale_factor : first generate a scale_factor*nside map, then decrease to nside, the larger the better.

    """

    lmax = 3 * (nside * scale_factor) -1 
    l = np.arange(1,lmax+1)
    N_white = (np.pi/(60*180) * nlev) ** 2

    if if_red:
        if n_red_P is not None:
            N_red_pol = (np.pi/(60*180) * n_red_P) ** 2
        else:
            N_red_pol = N_white
        N_red_tem = N_red_T
    else:
        N_red_pol = 0
        N_red_tem = 0
        l_knee_P = 1
        l_knee_T = 1
        alpha_knee_P = 1
        alpha_knee_T = 1

    
    N_noise_pol = 2 * (N_red_pol * (l/l_knee_P)**alpha_knee_P + N_white)
    N_noise_tem = (N_red_tem * (l/l_knee_T)**alpha_knee_T + N_white)

    np.random.seed(seed)
    noise_map_T = hp.synfast(N_noise_tem, nside * scale_factor, pol=False, new=True)
    noise_map_T = hp.ud_grade(noise_map_T, nside)
    #alm = curvedsky.utils.gauss1alm(lmax, N_noise_tem)
    #noise_map_T = curvedsky.utils.hp_alm2map(nside, lmax, lmax, alm)

    np.random.seed(seed+1234)
    noise_map_Q = hp.synfast(N_noise_pol, nside * scale_factor, pol=False, new=True)
    noise_map_Q = hp.ud_grade(noise_map_Q, nside)

    np.random.seed(seed+4321)
    noise_map_U = hp.synfast(N_noise_pol, nside * scale_factor, pol=False, new=True)
    noise_map_U = hp.ud_grade(noise_map_U, nside)

    return [noise_map_T, noise_map_Q, noise_map_U]


def call_noise_model(nside, noise_model, nlev=0, seed=0, N_red_T=None, l_knee_T=None, alpha_knee_T=None, n_red_P=None, l_knee_P=None, alpha_knee_P=None, if_red=False, scale_factor = 2):
    """
    Call different noise models to generate noise maps.

    * nside : nside
    * noise_model : a str the noise model parameters. 
                    (1) 'pixel_based' : generate Gaussian noise based on pixel-based noise RMS(i.e. sigma_pix) map (in uK-pix).
                                        Once you choose this model, 'nlev' should be a RMS map (in uK-pix).

                    (2) 'homogeneous' : generate homogeneous Gaussian noise based on a given noise level (in uK-arcmin).
                                        Once you choose this model, 'nlev' should be a float (in uK-arcmin).

                    (3) 'power_based' : generate 1/f noise and white noise based on noise power decribed by models with given parameters.
                                        Once you choose this model, you should specify the following parameters:
                                        * nlev :         noise level of white noise in uK-arcmin.
                                        * N_red_T :      1/f noise power for temperature, in uK^2-s
                                        * l_knee_T :     Specific scale where white noise becomes dominant.
                                        * alpha_knee_T : Power index control the decay of 1/f noise .
                                        * n_red_P :      1/f noise level for polarization, in uK-arcmin. If None, it will be set to nlev.
                                        * l_knee_P :     Specific scale where white noise becomes dominant.
                                        * alpha_knee_P : Power index control the decay of 1/f noise .
                                        * if_red :       If True, include 1/f noise, otherwise only white noise.
                                        * scale_factor : first generate a scale_factor*nside map, then decrease to nside, the larger the better.

    * seed : random seed

    Notice: the generated noise maps are 'TQU' in uK_CMB.

    The first two models (both draw smaple for map pixel) are more realistic and faster, while the last one (draw Gaussian field in Spherical Harmonic space) is more flexible but slower.

    """
    
    if noise_model == 'pixel_based':
        print("Generating noise based on sigma_pix map !!!")
        noise_map = noise(nlev, nside, seed=seed)
    elif noise_model == 'homogeneous':
        print("Generating homogenous noise based on nlev value !!!")
        noise_map = homo_noise(nlev, nside, seed=seed)
    elif noise_model == 'power_based':
        print("Generating total noise(1/f and white noise) based on noise power spectra !!!")
        noise_map = obs_noise(nlev, nside, seed, N_red_T, l_knee_T, alpha_knee_T, n_red_P, l_knee_P, alpha_knee_P, if_red, scale_factor)
    else:
        print("Please select a given noise model !!!")

    return noise_map