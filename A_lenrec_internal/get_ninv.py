import os
import sys
import numpy as np
import healpy as hp
import lenspyx
import multiprocessing as mp
import argparse

sys.path.insert(0, './')
from one import *
from utils import uKamin2uKpix, bl_eft, bl, apodize_mask
from generate_map import *


def ninv_NEW(nlev, savePath, fwhm_f=None):
    """ Noise inveres pixel variance for inhomogeneous filtering.

        * nlev      : it dependes ...

        All AliCPT's detectors are polarized and thus for simplicity
        nlev_Q = nlev_T * sqrt(2)
    """
    assert isinstance(nlev, list)
    nrms_f = hp.read_map(nlev[0], field=(0,1))
    fname_ninv_t = 'ninv_t.fits'
    fname_ninv_p = 'ninv_p.fits'


    # TODO Gaussian smoothing the variance map

    ninv_t = nrms_f[0] ** 2
    ninv_t[ninv_t!=0] = ninv_t[ninv_t!=0] ** -1  #逆方差
    ninv_p = nrms_f[1] ** 2
    ninv_p[ninv_p!=0] = ninv_p[ninv_p!=0] ** -1
        

    hp.write_map(os.path.join( savePath, fname_ninv_t ), ninv_t, overwrite=True)
    hp.write_map(os.path.join( savePath, fname_ninv_p ), ninv_p, overwrite=True)


ninv_NEW(nlev, savePath_ninv, fwhm_f=fwhm_f)  
