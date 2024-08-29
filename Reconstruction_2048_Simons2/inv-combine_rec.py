import sys
sys.path.insert(0, '../params')
import params as parfile   
from library_parameter import *
import healpy as hp
import numpy as np
import pylab as pl
from plancklens import utils

wiener1 = np.loadtxt("/sharefs/alicpt/users/chenwz/reconstruction_2048_simons/wiener.dat")  # L, NL, CL+NL
wiener2 = np.loadtxt("/sharefs/alicpt/users/chenwz/reconstruction_2048_simons2/wiener.dat")  # L, NL, CL+NL

ninv1 = 1. * utils.cli(wiener1[1]) 
ninv2 = 1. * utils.cli(wiener2[1]) 

weight1 = ninv1 * utils.cli(ninv1 + ninv2)
weight2 = ninv2 * utils.cli(ninv1 + ninv2)

nlpp_com = weight1**2 * wiener1[1] + weight2**2 * wiener2[1] 
np.savetxt(f"/sharefs/alicpt/users/chenwz/reconstruction_2048_simons2/nlpp_com.dat", [wiener1[0], nlpp_com, wiener1[2]-wiener1[1]+nlpp_com],)  # L, NL, CL+NL

for idx in range(0,200):
    rec_alm1 = hp.read_alm(f"/sharefs/alicpt/users/chenwz/reconstruction_2048_simons/rec_qlm/rec_alm_{idx}.fits")
    rec_alm2 = hp.read_alm(f"/sharefs/alicpt/users/chenwz/reconstruction_2048_simons2/rec_qlm/rec_alm_{idx}.fits")
    rec_alm = hp.almxfl(rec_alm1,weight1) + hp.almxfl(rec_alm2,weight2)
    hp.write_alm(f"/sharefs/alicpt/users/chenwz/reconstruction_2048_simons2/com_rec/rec_qlm_{idx}.fits", rec_alm, overwrite=True)
    print(f"rec_map_{idx}.fits saved")