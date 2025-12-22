""" ALL Plot Scripts

    * recon_cl plot
    * SNR plot
    * reconstruction map
写在shell中,这只是作为其中一个输入,需结合其他一起写入,利用mpi计算得到结果,现在用plot.py也可顺序计算得到结果。
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


# Error boxes   ：这个函数用于创建带有错误框的散点图,一个演示例子见WSL的/home/rabbit/workarea/1st work/2nd_ed/3rd_ed.ipynb
def make_error_boxes(ax, xdata, ydata, xerr, yerr, facecolor, edgecolor='None', alpha=0.5):
    errorboxes = []
    for x, y, xe, ye in zip(xdata, ydata, xerr.T, yerr.T):
        rect = Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())    #创建一个矩形（Rectangle）对象表示误差框：((左下角x,左下角y),宽度,高度)
        errorboxes.append(rect)
    # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor)#使用PatchCollection创建一个包含所有误差框的图形集合
    # PatchCollection 允许您将多个图形对象集中在一起,并在图形中一起显示它们。

    # Add collection to axes
    ax.add_collection(pc)   #将其添加到指定的轴（ax）





# Parameters
#savePath = './'
#qe_keys = {
#           'p_eb':['k', 'EB'],
#           'p_te':['r', 'TE']
#           } # MV estimator
#btype = 'agr2'

# Parameters
savePath = './'      #保存文件的路径,设置为当前目录 './'
qe_keys = {   #键值对：估计量类型：[颜色,'类型描述']
            'p':['royalblue', 'TT'] # Temperature only
           #,'p_p':['green', 'Pol only'] # Polarization only
#           'ptt':['tomato', 'MV'], # MV estimator
#          'pee':['royalblue', 'EE']
 #          ,'p_p':['green', 'Pol']
 #          ,'p_eb':['tomato', 'EB']
          } 
btype = 'agr2'
# 定义见plancklens的bandpowers.py,给定了binning的上下限,根据不同研究用途决定分bin. 比如研究宇宙学的参数的时候,只需要做L<400,而lensing一般就是1000到2000最大。
# 这里是lensing reconstruct。lensing还可以研究去透镜、算宇宙学参数等,我们就要用不同的L的区域和范围,那我们肯定要在这个范围里面尽量的让它显示的好看一点,主要是为了让出图好看。


# Plot Reconstructed power spectrum
fig, ax = plt.subplots()

for key in qe_keys.keys():  #ptt, p_p, p
    print(key)
    bp = bandpowers.ffp10_binner(key, key, par, btype, ksource='p')  #估计器&binning
    bells = bp.bin_lavs
    #bpower = (bp.get_dat_bandpowers() - bp.get_rdn0() - bp.get_n1()) * bp.get_bmmc()   #得到无偏功率谱估计量, 其中get_bmmc(): multiplicative MC correction
    bpower = (bp.get_dat_bandpowers() - bp.get_rdn0() ) * bp.get_bmmc()   #得到无偏功率谱估计量, 其中get_bmmc(): multiplicative MC correction
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
    print('SNR of', qe_key, 'is', np.dot(np.dot(signal, np.matrix(cov).I), signal)[0,0] ** 0.5)   #Fisher method, SNR=sqrt(C*Cov^-1*C),见Liu Jinyi文章(3.15)式，在存在foreground等因素的影响下相比下式更精确.这里[0,0]是因为点积结果为[[SNR]] 
    print('SNR of', qe_key, 'is', np.sqrt(sum(signal**2 / cov.diagonal())))    #误差传递公式，仅考虑auto-correlation(即对角元素,即方差)
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
#对角线应该相比非对角线大数量级，且加入的条件越少非对角线越小。
#不同L之间的d应该是正常来说应该是独立的，因而非对角线较小。由于foreground的不同l之间有联系，加入后导致非对角线增大（covariance直接线性相加）。
#各个L之间是这个比较独立，因为mode 与 mode之间的这个所谓的correlation是相当小的。如果它要它要大的话，这个宇宙就趋于无序了
    

'''
# Reconstruction map
def view_map(m, title, savePath, min=None, max=None, cmap='YlGnBu_r'):
    """ View map.
    """
    # TODO beautify this plot
    rot = [180, 60, 0]


    m = hp.read_map(m, verbose=False) if isinstance(m, str) else m
    m[ m==0. ] = np.nan # in case the input map is an apodization mask

    if min==None: min = m[ ~np.isnan(m) ].min()
    if max==None: max = m[ ~np.isnan(m) ].max()

    hp.orthview(m, title=title, min=min, max=max, rot=rot, half_sky=True, cmap=cmap)  #Plot a healpix map (given as an array) in Orthographic projection.
    hp.graticule()  #添加经纬度网格
    plt.savefig(savePath, dpi=300)


mask_b = np.where(hp.read_map(os.path.join(ALILENS, 'sims/ninv/ninv_t.fits')) > 0, 1, 0)  #读取一个特定文件,生成一个二进制掩码,其中大于零的值被标记为1,否则标记为0
lmax = 2048                                                                               #np.where(condition, "A", "B")满足条件的元素替换为A,不满足的取B
q2k = lambda l: l*(l + 1) / 2 # potential -> convergence   公式：kappa_lm = l(l+1)/2 * phi_lm
q2d = lambda l: (l*(l + 1)) ** 0.5 # potential -> deflection   公式：d_lm = sqrt(l(l+1)) * phi_lm
cut = np.where((np.arange(lmax + 1) > 8) * (np.arange(lmax + 1) < 2000), 1, 0) # band limit  #(np.arange(lmax + 1) > 8)和(np.arange(lmax + 1) < 2000)是bool数组

'''
#由PLANCK2015 (6)式的叙述,potential估计量有very red power spectrum,大部分功率都在大尺度,因此进行map cutting时会导致leakage问题。
#而convergence估计量有much white power spectrum 和 noise,尤其在大尺度上,因此下面的处理对kappa进行
'''

# wiener filter
wiener_dat = np.loadtxt(os.path.join(ALILENS, 'products/COM_Lensing_Inhf_2048_R1/MV/nlkk.dat')).transpose() #nlkk.dat存储格式: (L, NL^(\kappa\kappa), CL^(\kappa\kappa) + NL^(\kappa\kappa))
wiener = (wiener_dat[2] - wiener_dat[1]) * utils.cli(wiener_dat[2])  #即 C/(C+N)  #cli: Pseudo-inverse for positive cl-arrays.

# input deflection map
qlm_input = hp.map2alm(hp.read_map(os.path.join(ALILENS, 'sims/cmbs/map_P_1024_0100.fits')))
dlm_input = hp.almxfl(qlm_input, cut * q2d(np.arange(lmax + 1)))  
dmap_input = hp.alm2map(dlm_input, nside=1024)

# reconstruction map
klm_recon = hp.read_alm(os.path.join(ALILENS, 'products/COM_Lensing_Inhf_2048_R1/MV/dat_klm.fits'))
dlm_recon = hp.almxfl(klm_recon, cut * utils.cli(q2k(np.arange(lmax + 1)))
                                     * q2d(np.arange(lmax + 1))
                                     * wiener)    #这里的Wiener filter见PLANCK2015 (5)式
dmap_recon = hp.alm2map(dlm_recon, nside=1024)


# plot
view_map(dmap_input * mask_b, '', 'dmap_input.png', min=-0.0024, max=0.0024)
view_map(dmap_recon * mask_b, '', 'dmap_recon.png', min=-0.0024, max=0.0024)
'''






















