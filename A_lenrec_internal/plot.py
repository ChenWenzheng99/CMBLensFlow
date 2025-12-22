""" ALL Plot Scripts

    * recon_cl plot
    * SNR plot
    * reconstruction map
"""


import os
import sys
import numpy as np
import healpy as hp

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle # Error boxes
from matplotlib.collections import PatchCollection # Error boxes
from plancklens import utils
from plancklens.qresp import get_response
from utils import matrixshow

sys.path.insert(0, './')
from one import *
import params as par
import bandpowers


# Error boxes   
def make_error_boxes(ax, xdata, ydata, xerr, yerr, facecolor, edgecolor='None', alpha=0.5):
    errorboxes = []
    for x, y, xe, ye in zip(xdata, ydata, xerr.T, yerr.T):
        rect = Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())    
        errorboxes.append(rect)
    # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor)
    
    # Add collection to axes
    ax.add_collection(pc)   



_CONFIG_FILE = os.environ.get("ALI_CONFIG", "config.yaml")
with open(_CONFIG_FILE, "r") as f:
    _cfg = yaml.safe_load(f)

# Parameters
#savePath = './'
#qe_keys = {
#           'p_eb':['k', 'EB'],
#           'p_te':['r', 'TE']
#           } # MV estimator
#btype = 'agr2'

# Parameters
savePath = './'      
qe_keys = {         # define colors and labels for different QE keys
            'p':['royalblue', 'TT'], # Temperature only
           'p_p':['green', 'Pol only'], # Polarization only
           'ptt':['tomato', 'MV'], # MV estimator
           } 
btype = 'agr2'



# Plot Reconstructed power spectrum
fig, ax = plt.subplots()

for key in [_cfg['qe']["lib_qe_keys"][0]]:  #ptt, p_p, p
    print(key)
    bp = bandpowers.ffp10_binner(key, key, par, btype, ksource='p')  
    bells = bp.bin_lavs
    #bpower = (bp.get_dat_bandpowers() - bp.get_rdn0() - bp.get_n1()) * bp.get_bmmc()   #get_bmmc(): multiplicative MC correction
    bpower = (bp.get_dat_bandpowers() - bp.get_rdn0() ) * bp.get_bmmc()   
#    bpower = (bp.get_dat_bandpowers() - bp.get_rdn0())
    #np.savetxt('bandpowers.txt',[bells, bp.get_dat_bandpowers(), bp.get_rdn0(),  bp.get_bmmc()])

    # Potential Estimator
    yerr = np.sqrt(bp.get_cov().diagonal())
    l, u, c = bandpowers.get_blbubc(btype)    #下边界,上边界,band中心
    xerr = np.stack([bp.bin_lavs - l, u - bp.bin_lavs + 1])    #xerr什么样的
    make_error_boxes(ax, bells, bpower, xerr, np.stack([yerr, yerr]), facecolor=qe_keys[key][0], edgecolor='None', alpha=0.2)   #plot error boxes
    ax.scatter(bells, bpower, label='$C_L^{\hat\phi\hat\phi}-\hat N_L^{(0)}$  (%s)' % qe_keys[key][1], c=qe_keys[key][0], s=13) #plot binned cl

    np.savetxt('bandpowers.txt',[bells, bp.get_dat_bandpowers(), bp.get_rdn0(),  bp.get_bmmc(), ])
    np.savetxt('xerror.txt',xerr,)
    np.savetxt('yerror.txt',yerr,)


# fiducial lensing power spectrum
ax.plot(np.arange(5001), bp.clkk_fid, label=r'$C_L^{\phi\phi, fid}$', c='k')
ax.semilogx()
ax.semilogy()
ax.set_title('Lensing Reconstruction', fontsize=12)
ax.set_xlabel(r'$L$', fontsize=9)
ax.set_ylabel(r'$10^7 L^2 (L + 1)^2 C_L^{\phi\phi}/2\pi$', fontsize=9)
ax.legend(loc='upper right', fontsize=7)
#ax.set_xlim(l.min(), u.max())
#ax.set_ylim(0.001, 3.)
fig.savefig(os.path.join(ALILENS, 'recon_cl.pdf'))
#fig.savefig('recon_cl.png', dpi=300)





# Plot SNR plot
fig, ax = plt.subplots()
for qe_key in qe_keys.keys():
    print(qe_key)
    bp = bandpowers.ffp10_binner(qe_key, qe_key, par, btype, ksource='p')
    cov = bp.get_cov()
    signal = (bp.get_dat_bandpowers() - bp.get_rdn0() - bp.get_n1()) * bp.get_bmmc() #bp.get_bmmc(): Binned multiplicative MC correction
#    signal = bp.get_fid_bandpowers()
    l, u, c = bandpowers.get_blbubc(btype)
    SNRs = signal / np.sqrt(cov.diagonal())    #信噪比定义

    ax.plot(bp.bin_lavs, SNRs, c=qe_keys[qe_key][0], label='SNR (%s)' % qe_key)
    #two method to calculate SNR
    print('SNR of', qe_key, 'is', np.dot(np.dot(signal, np.matrix(cov).I), signal)[0,0] ** 0.5)   #Fisher method, SNR=sqrt(C*Cov^-1*C), more accurate
    print('SNR of', qe_key, 'is', np.sqrt(sum(signal**2 / cov.diagonal())))     # Error propagation formula, only considering auto-correlation (diagonal elements, i.e., variance)
#    print(cov)
ax.semilogx()
ax.set_title('Reconstruction SNR', fontsize=12)
ax.set_xlabel(r'$L$', fontsize=12)
ax.set_ylabel('SNR', fontsize=12)
ax.legend()
ax.set_xlim(l.min(), 2048)

fig.savefig(os.path.join(ALILENS, 'recon_snr.pdf'))

#plot covariance matrix
fig, ax = plt.subplots()
for qe_key in qe_keys.keys():
    print(qe_key)
    bp = bandpowers.ffp10_binner(qe_key, qe_key, par, btype, ksource='p')
    cov = bp.get_cov()
    matrixshow(cov,os.path.join(ALILENS, f'cov_matrix_{qe_key}.pdf'))

    


