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

        * nlev      : it depends ...,输入T的,单位muK-arcmin
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