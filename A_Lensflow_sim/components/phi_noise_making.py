# external
import numpy as np
from matplotlib.pyplot import *
import camb
# from cmblensplus/wrap/
import basic
import curvedsky as cs
# from cmblensplus/utils/
import healpy as hp

from lenspyx.utils import camb_clfile
from lenspyx.utils_hp import synalm, almxfl, alm2cl

import sys
sys.path.append('/root/download/cmblensplus2/utils')

import plottools as pl
import cmb

def cli(cl):
    """Pseudo-inverse for positive cl-arrays.

    """
    ret = np.zeros_like(cl)
    ret[np.where(cl != 0)] = 1. / cl[np.where(cl != 0)]
    return ret

def gaussian_phi_n0(lmin_cut,lmax_cut,rlmin_cut=[0],rlmax_cut=[0],nside=2048,sigmas=[0],thetas=[0],flag=5,seed=1234):
   """
   nside:一般默认为2048
   flag: 0,1,2,3,4,5,依次为TT,TE,EE,TB,EB,MV 的 N0
   sigma:white noise level, 单位为uK-arcmin
   theta:FWHM,单位为arcmin
   rlmin_cut,rlmax_cut: CMB multipole range for reconstruction
   Lmin_cut,Lmax_cut: maximum multipole of output normalization
   """

   Tcmb  = 2.726e6    # CMB temperature
   #nside = 2048
   lmax = 3*nside-1
   Lmin, Lmax  = 2, 6900       # maximum multipole of output normalization
   rlmin, rlmax = 2, 6900  # CMB multipole range for reconstruction

   L = np.linspace(0,Lmax,Lmax+1)
   Lfac = (L*(L+1.))**2/(2*np.pi)
   ac2rad = np.pi/10800.
   lTmax = rlmax_cut
   quad = ['TT','TE','EE','TB','EB','MV']
   QDO = [True,True,True,True,True,False] # this means that TT, TE, EE, TB and EB are used for MV estimator
   Tcmb = 2.726e6    # CMB temperature

   #from cmblensplus
   # ucl is an array of shape [0:5,0:rlmax+1] and ucl[0,:] = TT, ucl[1,:] = EE, ucl[2,:] = TE, lcl[3,:] = phiphi, lcl[4,:] = Tphi
   #ucl_clps = cmb.read_camb_cls('/root/download/cmblensplus2/example/data/unlensedcls.dat',ftype='scal',output='array')[:,:rlmax+1]
   # lcl is an array of shape [0:4,0:rlmax+1] and lcl[0,:] = TT, lcl[1,:] = EE, lcl[2,:] = BB, and lcl[3,:] = TE
   #lcl_clps = cmb.read_camb_cls('/root/download/cmblensplus2/example/data/lensedcls.dat',ftype='lens',output='array')[:,:rlmax+1]

   #from Planck FFP10
   cl1 = camb_clfile('/root/download/plancklens/plancklens/data/cls/FFP10_wdipole_lenspotentialCls.dat',rlmax)   #TT,EE,BB,TE,PP,TP,EP
   cl2 = camb_clfile('/root/download/plancklens/plancklens/data/cls/FFP10_wdipole_lensedCls.dat',rlmax)            #TT,EE,BB,TE
   ucl = np.zeros((5,rlmax+1))
   lcl = np.zeros((4,rlmax+1))
   ucl[3,:] = cl1['pp'][:rlmax+1] 
   lcl[0,:] = cl2['tt'][:rlmax+1] / Tcmb**2
   lcl[1,:] = cl2['ee'][:rlmax+1] / Tcmb**2
   lcl[2,:] = cl2['bb'][:rlmax+1] / Tcmb**2
   lcl[3,:] = cl2['te'][:rlmax+1] / Tcmb**2

   thetas     #FWHM
   sigmas     #Noise levels
   Ag = {}
   Ac = {}

   # 初始化一个形状为 (len(thetas), 4, rlmax+1) 的数组，用于存储每次循环生成的噪声谱
   nl_list = np.zeros((len(thetas), 4, rlmax + 1))
   lcl_list = np.zeros((len(thetas), 4, rlmax + 1))
   ocl_list = np.zeros((len(thetas), 4, rlmax + 1))
   # 对每一组 (theta, sigma) 进行循环
   for i, (theta, sigma) in enumerate(zip(thetas, sigmas)):
      # 创建一个形状为 (4, rlmax+1) 的空数组 nl
      if sigma == 0:
         sigma = 1e-50
      nl = np.zeros((4, rlmax + 1))
      nl[0, :] = 0.5 * (sigma * ac2rad / Tcmb)**2 * np.exp(L * (L + 1.) * (theta * ac2rad)**2 / np.log(2.) / 8.)
      nl[1, :] = 2. * nl[0, :]
      nl[2, :] = 2. * nl[0, :]
      
      # 将当前 nl 存入 nl_list 中
      nl_list[i, :, :] = nl

   rlow = np.max(rlmin_cut)
   rhigh = np.min(rlmax_cut)

   # 计算逆方差
   inv_variances = 1.0 *cli(nl_list)

   # 计算线性组合系数
   weights = inv_variances * cli(np.sum(inv_variances, axis=0))
   weights[:,:,:rlow] = 0
   weights[:,:,rhigh+1:] = 0

   anti_weights = np.where(weights != 0, 0, 1)
   for i, (rlmin, rlmax) in enumerate(zip(rlmin_cut, rlmax_cut)):
      anti_weights[i, :, :rlmin] = 0
      anti_weights[i, :, rlmax+1:] = 0

   # 对每个频道进行线性组合
   result_nl = np.sum(weights * nl_list, axis=0) + np.sum(anti_weights * nl_list, axis=0)

   ocl = lcl + result_nl

   rlmin_cut = np.min(rlmin_cut)
   rlmax_cut = np.max(rlmax_cut)
   Ag[0], Ac[0], Wg, Wc = cs.norm_quad.qall('lens',QDO,Lmax,rlmin_cut,rlmax_cut,lcl[:,:rlmax_cut+1],ocl[:,:rlmax_cut+1],)  #RAW NORMALIZATION(未乘Lfac)

   pl.plot_1dstyle(fsize=[7,4],xmin=2,xmax=Lmax,xlog=True,ymin=1e-9,ymax=2e-6,ylog=True,ylab=r'$L^2(L+1)^2C^{\phi\phi}_L/2\pi$')

   dlpp  = L**2*(L+1.)**2/(2*np.pi)*ucl[3,:]

   for qi, q in enumerate(quad):
      plot(L,Lfac*Ag[0][qi,:],ls='-',label=q)
      #plot(L,Ag[1][qi,:],ls='--')

   plot(L,dlpp,ls='-',label='$\phi$')
   legend()
   np.random.seed(seed)
   #alm = cs.utils.gauss1alm(Lmax, Ag[0][flag,:])  #依次为TT,TE,EE,TB,EB,MV 的 N0
   #phi_N0_map = cs.utils.hp_alm2map(nside, Lmax, Lmax, alm)  #seed不生效

   #alm = hp.synalm(Ag[0][flag,:], lmax=Lmax, mmax=Lmax)
   phi_N0_map = hp.synfast(Ag[0][flag,:lmax_cut], nside, lmax=lmax_cut)  #seed可以生效

   hp.mollview(phi_N0_map)
   return phi_N0_map

def gaussian_phi_n0_from_map(map,rlmin_cut,rlmax_cut,lmin_cut,lmax_cut,nside=2048,sigmas=[0],thetas=[0],flag=5,seed=1234):
   """
   map: T,Q,U,P map
   nside:一般默认为2048
   flag: 0,1,2,3,4,5,依次为TT,TE,EE,TB,EB,MV 的 N0
   sigma:white noise level, 单位为uK-arcmin
   theta:FWHM,单位为arcmin
   rlmin_cut,rlmax_cut: CMB multipole range for reconstruction
   Lmin_cut,Lmax_cut: maximum multipole of output normalization
   """
   assert rlmax_cut <= 3*nside-1 and lmax_cut <= 3*nside-1
   Tcmb  = 2.726e6    # CMB temperature
   #nside = 2048
   lmax = 3*nside-1
   Lmin, Lmax  = 2, 3*nside-1       # maximum multipole of output normalization
   rlmin, rlmax = 2, 3*nside-1  # CMB multipole range for reconstruction

   L = np.linspace(0,Lmax,Lmax+1)
   Lfac = (L*(L+1.))**2/(2*np.pi)
   ac2rad = np.pi/10800.
   lTmax = rlmax_cut
   quad = ['TT','TE','EE','TB','EB','MV']
   QDO = [True,True,True,True,True,False] # this means that TT, TE, EE, TB and EB are used for MV estimator
   Tcmb = 2.726e6    # CMB temperature
   
   #from cmblensplus
   # ucl is an array of shape [0:5,0:rlmax+1] and ucl[0,:] = TT, ucl[1,:] = EE, ucl[2,:] = TE, lcl[3,:] = phiphi, lcl[4,:] = Tphi
   #ucl_clps = cmb.read_camb_cls('/root/download/cmblensplus2/example/data/unlensedcls.dat',ftype='scal',output='array')[:,:rlmax+1]
   # lcl is an array of shape [0:4,0:rlmax+1] and lcl[0,:] = TT, lcl[1,:] = EE, lcl[2,:] = BB, and lcl[3,:] = TE
   #lcl_clps = cmb.read_camb_cls('/root/download/cmblensplus2/example/data/lensedcls.dat',ftype='lens',output='array')[:,:rlmax+1]

   #from Planck FFP10
   cl = hp.anafast([map[0],map[1],map[2]],lmax=rlmax)
   cl1 = camb_clfile('/root/download/plancklens/plancklens/data/cls/FFP10_wdipole_lenspotentialCls.dat',rlmax)   #TT,EE,BB,TE,PP,TP,EP
   cl2 = camb_clfile('/root/download/plancklens/plancklens/data/cls/FFP10_wdipole_lensedCls.dat',rlmax)            #TT,EE,BB,TE
   ucl = np.zeros((5,rlmax+1))
   lcl = np.zeros((4,rlmax+1))
   ucl[3,:] = hp.anafast(map[3],lmax=rlmax)
   lcl[0,:] = cl[0]
   lcl[1,:] = cl[1]
   lcl[2,:] = cl[2]
   lcl[3,:] = cl[3]

   thetas     #FWHM
   sigmas     #Noise levels
   Ag = {}
   Ac = {}

   # 初始化一个形状为 (len(thetas), 4, rlmax+1) 的数组，用于存储每次循环生成的噪声谱
   nl_list = np.zeros((len(thetas), 4, rlmax + 1))
   # 对每一组 (theta, sigma) 进行循环
   for i, (theta, sigma) in enumerate(zip(thetas, sigmas)):
      # 创建一个形状为 (4, rlmax+1) 的空数组 nl
      nl = np.zeros((4, rlmax + 1))
      nl[0, :] = 0.5 * (sigma * ac2rad / Tcmb)**2 * np.exp(L * (L + 1.) * (theta * ac2rad)**2 / np.log(2.) / 8.)
      nl[1, :] = 2. * nl[0, :]
      nl[2, :] = 2. * nl[0, :]
      nl[0, lTmax+1:] = 1e30
      # 将当前 nl 存入 nl_list 中
      nl_list[i, :, :] = nl
   # 计算逆方差
   inv_variances = 1.0 *cli(nl_list) 

   # 计算线性组合系数
   weights = inv_variances * cli(np.sum(inv_variances, axis=0))

   # 对每个频道进行线性组合
   result_nl = np.sum(weights * nl_list, axis=0)

   ocl = lcl + result_nl
   Ag[0], Ac[0], Wg, Wc = cs.norm_quad.qall('lens',QDO,Lmax,rlmin_cut,rlmax_cut,lcl[:,:rlmax_cut+1],ocl[:,:rlmax_cut+1],)  #RAW NORMALIZATION(未乘Lfac)

   pl.plot_1dstyle(fsize=[7,4],xmin=2,xmax=Lmax,xlog=True,ymin=1e-9,ymax=2e-6,ylog=True,ylab=r'$L^2(L+1)^2C^{\phi\phi}_L/2\pi$')

   dlpp  = L**2*(L+1.)**2/(2*np.pi)*ucl[3,:]

   for qi, q in enumerate(quad):
      plot(L,Lfac*Ag[0][qi,:],ls='-',label=q)
      #plot(L,Ag[1][qi,:],ls='--')

   plot(L,dlpp,ls='-',label='$\phi$')
   legend()
   np.random.seed(seed)
   #alm = cs.utils.gauss1alm(Lmax, Ag[0][flag,:])  #依次为TT,TE,EE,TB,EB,MV 的 N0
   #phi_N0_map = cs.utils.hp_alm2map(nside, Lmax, Lmax, alm)  #seed不生效

   #alm = hp.synalm(Ag[0][flag,:], lmax=Lmax, mmax=Lmax)
   phi_N0_map = hp.synfast(Ag[0][flag,:lmax_cut], nside, lmax=lmax_cut)  #seed可以生效

   hp.mollview(phi_N0_map)
   return phi_N0_map

def gaussian_phi_n0_from_cl(cl,clpp,rlmin_cut,rlmax_cut,lmin_cut,lmax_cut,nside=2048,sigmas=[0],thetas=[0],flag=5,seed=1234):
   """
   map: T,Q,U,P map
   nside:一般默认为2048
   flag: 0,1,2,3,4,5,依次为TT,TE,EE,TB,EB,MV 的 N0
   sigma:white noise level, 单位为uK-arcmin
   theta:FWHM,单位为arcmin
   rlmin_cut,rlmax_cut: CMB multipole range for reconstruction
   Lmin_cut,Lmax_cut: maximum multipole of output normalization
   """
   assert rlmax_cut <= 3*nside-1 and lmax_cut <= 3*nside-1
   Tcmb  = 2.726e6    # CMB temperature
   #nside = 2048
   lmax = 3*nside-1
   Lmin, Lmax  = 2, 3*nside-1       # maximum multipole of output normalization
   rlmin, rlmax = 2, 3*nside-1  # CMB multipole range for reconstruction

   L = np.linspace(0,Lmax,Lmax+1)
   Lfac = (L*(L+1.))**2/(2*np.pi)
   ac2rad = np.pi/10800.
   lTmax = rlmax_cut
   quad = ['TT','TE','EE','TB','EB','MV']
   QDO = [True,True,True,True,True,False] # this means that TT, TE, EE, TB and EB are used for MV estimator
   Tcmb = 2.726e6    # CMB temperature
   
   #from cmblensplus
   # ucl is an array of shape [0:5,0:rlmax+1] and ucl[0,:] = TT, ucl[1,:] = EE, ucl[2,:] = TE, lcl[3,:] = phiphi, lcl[4,:] = Tphi
   #ucl_clps = cmb.read_camb_cls('/root/download/cmblensplus2/example/data/unlensedcls.dat',ftype='scal',output='array')[:,:rlmax+1]
   # lcl is an array of shape [0:4,0:rlmax+1] and lcl[0,:] = TT, lcl[1,:] = EE, lcl[2,:] = BB, and lcl[3,:] = TE
   #lcl_clps = cmb.read_camb_cls('/root/download/cmblensplus2/example/data/lensedcls.dat',ftype='lens',output='array')[:,:rlmax+1]

   #from Planck FFP10
   #cl = hp.anafast([map[0],map[1],map[2]])
   #cl1 = camb_clfile('/root/download/plancklens/plancklens/data/cls/FFP10_wdipole_lenspotentialCls.dat',rlmax)   #TT,EE,BB,TE,PP,TP,EP
   #cl2 = camb_clfile('/root/download/plancklens/plancklens/data/cls/FFP10_wdipole_lensedCls.dat',rlmax)            #TT,EE,BB,TE
   ucl = np.zeros((5,rlmax+1))
   lcl = np.zeros((4,rlmax+1))
   ucl[3,:] = clpp[:rlmax+1]
   lcl[0,:] = cl[:rlmax+1,0]/ Tcmb**2
   lcl[1,:] = cl[:rlmax+1,1]/ Tcmb**2
   lcl[2,:] = cl[:rlmax+1,2]/ Tcmb**2
   lcl[3,:] = cl[:rlmax+1,3]/ Tcmb**2

   thetas     #FWHM
   sigmas     #Noise levels
   Ag = {}
   Ac = {}

   # 初始化一个形状为 (len(thetas), 4, rlmax+1) 的数组，用于存储每次循环生成的噪声谱
   nl_list = np.zeros((len(thetas), 4, rlmax + 1))
   # 对每一组 (theta, sigma) 进行循环
   for i, (theta, sigma) in enumerate(zip(thetas, sigmas)):
      # 创建一个形状为 (4, rlmax+1) 的空数组 nl
      nl = np.zeros((4, rlmax + 1))
      nl[0, :] = 0.5 * (sigma * ac2rad / Tcmb)**2 * np.exp(L * (L + 1.) * (theta * ac2rad)**2 / np.log(2.) / 8.)
      nl[1, :] = 2. * nl[0, :]
      nl[2, :] = 2. * nl[0, :]
      nl[0, lTmax+1:] = 1e30
      # 将当前 nl 存入 nl_list 中
      nl_list[i, :, :] = nl
   # 计算逆方差
   inv_variances = 1.0 *cli(nl_list) 

   # 计算线性组合系数
   weights = inv_variances * cli(np.sum(inv_variances, axis=0))

   # 对每个频道进行线性组合
   result_nl = np.sum(weights * nl_list, axis=0)

   ocl = lcl + result_nl
   Ag[0], Ac[0], Wg, Wc = cs.norm_quad.qall('lens',QDO,Lmax,rlmin_cut,rlmax_cut,lcl[:,:rlmax_cut+1],ocl[:,:rlmax_cut+1],)  #RAW NORMALIZATION(未乘Lfac)

   pl.plot_1dstyle(fsize=[7,4],xmin=2,xmax=Lmax,xlog=True,ymin=1e-9,ymax=2e-6,ylog=True,ylab=r'$L^2(L+1)^2C^{\phi\phi}_L/2\pi$')

   dlpp  = L**2*(L+1.)**2/(2*np.pi)*ucl[3,:]

   for qi, q in enumerate(quad):
      plot(L,Lfac*Ag[0][qi,:],ls='-',label=q)
      #plot(L,Ag[1][qi,:],ls='--')

   plot(L,dlpp,ls='-',label='$\phi$')
   legend()
   np.random.seed(seed)
   #alm = cs.utils.gauss1alm(Lmax, Ag[0][flag,:])  #依次为TT,TE,EE,TB,EB,MV 的 N0
   #phi_N0_map = cs.utils.hp_alm2map(nside, Lmax, Lmax, alm)  #seed不生效

   #alm = hp.synalm(Ag[0][flag,:], lmax=Lmax, mmax=Lmax)
   phi_N0_map = hp.synfast(Ag[0][flag,:lmax_cut], nside, lmax=lmax_cut)  #seed可以生效

   hp.mollview(phi_N0_map)
   return phi_N0_map

'''

def gaussian_phi_n0(nside, sigma, theta, flag=5,  ):
    """
      nside:一般默认为2048
      flag: 0,1,2,3,4,5,依次为TT,TE,EE,TB,EB,MV 的 N0
      sigma:white noise level, 单位为uK-arcmin
      theta:FWHM,单位为arcmin
    """
    import numpy as np
    import camb
    # from cmblensplus/wrap/
    import basic
    import curvedsky as cs
    # from cmblensplus/utils/
    import healpy as hp

    import sys
    sys.path.append('/root/download/cmblensplus2/utils')

    import plottools as plot
    import cmb
    Tcmb  = 2.726e6    # CMB temperature
    #nside = 2048
    lmax = 3*nside-1
    Lmin, Lmax  = 2, 4999       # maximum multipole of output normalization
    rlmin, rlmax = 2, 4999  # CMB multipole range for reconstruction
    L = np.linspace(0,Lmax,Lmax+1)
    Lfac = (L*(L+1.))**2/(2*np.pi)
    ac2rad = np.pi/10800.
    lTmax = rlmax
    quad = ['TT','TE','EE','TB','EB','MV']
    QDO = [True,True,True,True,True,False] # this means that TT, TE, EE, TB and EB are used for MV estimator

    # ucl is an array of shape [0:5,0:rlmax+1] and ucl[0,:] = TT, ucl[1,:] = EE, ucl[2,:] = TE, lcl[3,:] = phiphi, lcl[4,:] = Tphi
    ucl = cmb.read_camb_cls('/root/download/cmblensplus2/example/data/unlensedcls.dat',ftype='scal',output='array')[:,:rlmax+1]
    # lcl is an array of shape [0:4,0:rlmax+1] and lcl[0,:] = TT, lcl[1,:] = EE, lcl[2,:] = BB, and lcl[3,:] = TE
    lcl = cmb.read_camb_cls('/root/download/cmblensplus2/example/data/lensedcls.dat',ftype='lens',output='array')[:,:rlmax+1]

    #from Planck FFP10
    cl1 = camb_clfile('/root/download/plancklens/plancklens/data/cls/FFP10_wdipole_lenspotentialCls.dat',rlmax)   #TT,EE,BB,TE,PP,TP,EP
    cl2 = camb_clfile('/root/download/plancklens/plancklens/data/cls/FFP10_wdipole_lensedCls.dat',rlmax)            #TT,EE,BB,TE
    ucl[3,:] = cl1['pp'][:rlmax+1]
    lcl[0,:] = cl2['tt'][:rlmax+1]
    lcl[1,:] = cl2['ee'][:rlmax+1]
    lcl[2,:] = cl2['bb'][:rlmax+1]
    lcl[3,:] = cl2['te'][:rlmax+1]


    thetas = [theta]      #FWHM
    sigmas = [sigma]      #Noise levels
    Ag = {}
    Ac = {}
    for i, (sig, theta) in enumerate(zip(sigmas,thetas)):
       nl  = np.zeros((4,rlmax+1))
       nl[0,:] = .5*(sig*ac2rad/Tcmb)**2*np.exp(L*(L+1.)*(theta*ac2rad)**2/np.log(2.)/8.)
       nl[1,:] = 2.*nl[0,:]
       nl[2,:] = 2.*nl[0,:]
       nl[0,lTmax+1:] = 1e30
       ocl = lcl + nl
       Ag[i], Ac[i], Wg, Wc = curvedsky.norm_quad.qall('lens',QDO,Lmax,rlmin,rlmax,lcl,ocl,)  #RAW NORMALIZATION(未乘Lfac)

    plottools.plot_1dstyle(fsize=[7,4],xmin=2,xmax=Lmax,xlog=True,ymin=1e-9,ymax=2e-6,ylog=True,ylab=r'$L^2(L+1)^2C^{\phi\phi}_L/2\pi$')

    dlpp  = L**2*(L+1.)**2/(2*np.pi)*ucl[3,:]

    for qi, q in enumerate(quad):
       plot(L,Lfac*Ag[0][qi,:],ls='-',label=q)
       #plot(L,Ag[1][qi,:],ls='--')

    plot(L,dlpp,ls='-',label='$\phi$')
    legend()
    close()
    alm = curvedsky.utils.gauss1alm(Lmax, Ag[0][flag,:])  #依次为TT,TE,EE,TB,EB,MV 的 N0
    phi_N0_map = curvedsky.utils.hp_alm2map(nside, Lmax, Lmax, alm)
    return phi_N0_map

'''