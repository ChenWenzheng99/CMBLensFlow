from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

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

sys.path.append('/sharefs/alicpt/users/chenwz/download/cmblensplus2/utils')
sys.path.append('/sharefs/alicpt/users/chenwz/Testarea/FORECAST/Foreground/foreground_run')

from lensingb_mine import *
from noise_making import *
from utils_mine import *
from phi_noise_making import *
from generate_map import *
from lenspyx.utils import camb_clfile
from lenspyx.utils_hp import synalm, almxfl, alm2cl

camb_path = os.path.realpath(os.path.join(os.getcwd(),'..'))    
sys.path.insert(0,camb_path)

import camb
from camb import model, initialpower ,correlations

from scipy.special import factorial

import pymaster as nm

def get_pseudo_cl(nside,mask_input,flag,bin,lmax=None,apo=False,f0=None,f2=None,is_Dell=True):
    """
        nside:
        mask_input:最好是nside的mask,否则apodization很慢;或者输入apodization后的mask
        flag: '00' '02' '22',依次为spin-0 x spin-0,spin-0 x spin-2,spin-2 x spin-2
        bin: 每个bin的多极项数目
        lmax: 用于计算pseudo-Cl的lmax,默认为-1,即使用3*nside-1
        apo: 是否输入了apodization后的mask
        f0: spin-0 field
        f2: spin-2 field
    """
        
    import numpy as np
    import healpy as hp
    import matplotlib.pyplot as plt

    # Import the NaMaster python wrapper
    import pymaster as nmt

    if lmax is None:
        lmax = 3*nside - 1
    if apo == True:
        mask = mask_input
    else:
        mask = nmt.mask_apodization(mask_input,  1., apotype="Smooth")           # apodization中的参数可以调整，这里为 1 degree
    #hp.orthview(mask, half_sky=True, rot=[180, 30, 0],cmap='YlGnBu_r',title='Apodized mask')
    print('mask apodization done')
    
    # Read healpix maps and initialize a spin-0 and spin-2 field
    
    # Initialize binning scheme with 'bin' ells per bandpower
    #b = nmt.NmtBin.from_nside_linear(nside, bin, is_Dell=is_Dell)
    b = nmt.NmtBin.from_lmax_linear(lmax, bin, is_Dell=is_Dell)
    ell_arr = b.get_effective_ells()
    if isinstance(f0, (list, tuple)):
        f0_1 = [f0[0]]
        f0_2 = [f0[1]]
    else:
        f0_1 = [f0]
        f0_2 = [f0]
    print(len(f0_1),len(f0_2))
    # Compute MASTER estimator
    # spin-0 x spin-0
    if flag == '00':
        f_01 = nmt.NmtField(mask, f0_1,)
        f_02 = nmt.NmtField(mask, f0_2,)

        print('pseudo cl calculating ')
        cl_00 = nmt.compute_full_master(f_01, f_02, b)    #TT为 cl_00[0]
        return ell_arr,cl_00

    # spin-0 x spin-2
    if flag == '02':
        f_0 = nmt.NmtField(mask, f0_1,)
        f_2 = nmt.NmtField(mask, f2, spin=2, purify_e=True, purify_b=True, n_iter_mask_purify=3, )
        assert f_0 is not None
        assert f_2 is not None
        print('pseudo cl calculating ')
        cl_02 = nmt.compute_full_master(f_0, f_2, b)
        return ell_arr,cl_02
    
    # spin-2 x spin-2
    if flag == '22':
        f_2 = nmt.NmtField(mask, f2, spin=2, purify_e=True, purify_b=True, n_iter_mask_purify=3, )
        assert f_2 is not None
        print('pseudo cl calculating ')
        cl_22 = nmt.compute_full_master(f_2, f_2, b)      #EE,BB 分别为 cl_22[0]，cl_22[3]
        return ell_arr,cl_22
    
def add_beam(fwhm_ac,nside,maps,lmax=None,pixwin=False):
    if lmax == None:
        lmax =3*nside-1
    transf = bl(fwhm_ac, nside=nside, lmax=lmax, pixwin=pixwin)
    tlm_len = hp.map2alm(maps[0], lmax=lmax)
    elm_len, blm_len = hp.map2alm_spin([maps[1], maps[2]], 2, lmax=lmax)
    Tlen = hp.alm2map(hp.almxfl(tlm_len, transf, inplace=True), nside, verbose=False)
    Qlen, Ulen = hp.alm2map_spin([hp.almxfl(elm_len, transf), hp.almxfl(blm_len, transf)], nside, 2, lmax)
    return Tlen,Qlen,Ulen

def map_cut(nside,maps,lmax):
    tlm_len = hp.map2alm(maps[0], lmax=lmax)
    elm_len, blm_len = hp.map2alm_spin([maps[1], maps[2]], 2, lmax=lmax)
    Tlen = hp.alm2map(tlm_len,nside, verbose=False)
    Qlen, Ulen = hp.alm2map_spin([elm_len, blm_len,], nside, 2, lmax)
    return Tlen,Qlen,Ulen


def decon_map(fwhm_ac,nside,maps,lmax,pixwin=False):
    """
    deconvolve map with beam
    """
    if lmax == None:
        lmax =3*nside-1
    transf = 1/bl(fwhm_ac, nside=nside, lmax=lmax, pixwin=pixwin)
    tlm_len = hp.map2alm(maps[0], lmax=lmax)
    elm_len, blm_len = hp.map2alm_spin([maps[1], maps[2]], 2, lmax=lmax)
    Tlen = hp.alm2map(hp.almxfl(tlm_len, transf, inplace=True), nside, verbose=False)
    Qlen, Ulen = hp.alm2map_spin([hp.almxfl(elm_len, transf), hp.almxfl(blm_len, transf)], nside, 2, lmax)
    return Tlen,Qlen,Ulen

def qu2eb_map(nside,lmax,Qmap,Umap):
    """convert Q/U map to E/B map, only correct for full sky, but leads negligible corruption to partial sky E mode"""
    elm,blm = hp.map2alm_spin([Qmap,Umap],2)
    Emap = hp.alm2map(elm,nside,)
    Bmap = hp.alm2map(blm,nside,)
    return Emap,Bmap

def bl(fwhm, lmax=None, nside=None, pixwin=True):    #bl包括相当于beam window function 和 pixel window function,相当于e^-(l(l+1)* \sigma^2)(这是beam window function)
    """ Transfer function.

        * fwhm      : beam fwhm in arcmin
        * lmax      : lmax
        * pixwin    : whether include pixwin in beam transfer function
        * nside     : nside
    """
    assert lmax or nside
    lmax = min( 3 * nside - 1, lmax ) if nside and lmax else lmax if lmax else 3*nside - 1
    ret = hp.gauss_beam(fwhm * np.pi / 60. / 180., lmax=lmax)   #return: beam window function
    if pixwin:
        assert nside is not None
        ret *= hp.pixwin(nside, lmax=lmax)      #hp.pixwin: Return the pixel window function for the given nside
    return ret

def replace_fwhm_alm(nside,alm,fwhm_old,fwhm_new,pix_old,pix_new,lmax):
    """
    replace old beam of alm with new beam in arcmin
    """
    bl_old = bl(fwhm_old, nside=nside, lmax=lmax, pixwin=pix_old)     #sim.py生成的有pixwin,sim_ali.py生成的没有
    bl_new = bl(fwhm_new, nside=nside, lmax=lmax, pixwin=pix_new)    #还原为没有beam和pixwin
    alm_deconv = hp.almxfl(alm,1/bl_old * bl_new)
    return alm_deconv

def replace_fwhm_map(nside,lmax,map,fwhm_old,fwhm_new,pix_old,pix_new,):
    """
    replace old beam of map with new beam in arcmin
    """
    alm = hp.map2alm(map)
    alm_deconv = replace_fwhm_alm(nside,alm,fwhm_old,fwhm_new,pix_old,pix_new,lmax)
    map_deconv = hp.alm2map(alm_deconv,nside)
    return map_deconv

##########################################################################
##########################    Load map dir     ###########################

# 获取传递给脚本的参数
args = sys.argv[1:]

# 打印当前作业的编号
# 计算当前作业的文件编号列表
file_indices = comm_rank + int(args[0])

# 打印当前作业的文件编号列表
print(f"Job {comm_rank} will process files: {file_indices}")

# map读取路径 
temp_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/FOREGROUND_FORECAST/output_temp_rec/TEMP_MAP_{file_indices}.fits"

CMB_2048_filename = f"/sharefs/alicpt/users/chenwz/reconstruction_2048_simons/test/ALILENS/sims/cmbs/map_TQU_2048_{file_indices:04d}.fits"                    #TQU CMB 2048
noise_27_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/ALL_MAP_2048/QUP_NOISE/SAT/27/QUP_N_{file_indices}.npy"     #QU NOISE
noise_39_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/ALL_MAP_2048/QUP_NOISE/SAT/39/QUP_N_{file_indices}.npy"     #QU NOISE       
noise_93_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/ALL_MAP_2048/QUP_NOISE/SAT/93/QUP_N_{file_indices}.npy"     #QU NOISE
noise_145_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/ALL_MAP_2048/QUP_NOISE/SAT/145/QUP_N_{file_indices}.npy"     #QU NOISE
noise_225_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/ALL_MAP_2048/QUP_NOISE/SAT/225/QUP_N_{file_indices}.npy"     #QU NOISE
noise_280_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/ALL_MAP_2048/QUP_NOISE/SAT/280/QUP_N_{file_indices}.npy"     #QU NOISE

LT_LT_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/FOREGROUND_FORECAST/post_cal_temp_rec_new3/LT_LT/dlbb_LT_LT_{file_indices}.txt"
obs_LT_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/FOREGROUND_FORECAST/post_cal_temp_rec_new3/LT_obs/dlbb_obs_LT_{file_indices}.pickle"
obs_obs_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/FOREGROUND_FORECAST/post_cal_temp_rec_new3/obs_obs/dlbb_obs_obs_{file_indices}.pickle"
noise_FG_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/FOREGROUND_FORECAST/post_cal_temp_rec_new3/noise&FG/dlbb_noise_FG_{file_indices}.pickle"
FG_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/FOREGROUND_FORECAST/post_cal_temp_rec_new3/FG_only/dlbb_FG_{file_indices}.pickle"

print(f"Job {comm_rank} will read template map from {temp_filename }:")
print(f"Job {comm_rank} will write map to {LT_LT_filename}:")

"""
FG_27 = hp.read_map('/sharefs/alicpt/users/chenwz/Testarea/FORECAST/Foreground/foreground_Simons/gaussian_toy_fg_27GHz.fits',field=(1,2))   #QU FG
FG_39 = hp.read_map('/sharefs/alicpt/users/chenwz/Testarea/FORECAST/Foreground/foreground_Simons/gaussian_toy_fg_39GHz.fits',field=(1,2))   #QU FG
FG_93 = hp.read_map('/sharefs/alicpt/users/chenwz/Testarea/FORECAST/Foreground/foreground_Simons/gaussian_toy_fg_93GHz.fits',field=(1,2))   #QU FG
FG_145 = hp.read_map('/sharefs/alicpt/users/chenwz/Testarea/FORECAST/Foreground/foreground_Simons/gaussian_toy_fg_145GHz.fits',field=(1,2))   #QU FG
FG_225 = hp.read_map('/sharefs/alicpt/users/chenwz/Testarea/FORECAST/Foreground/foreground_Simons/gaussian_toy_fg_225GHz.fits',field=(1,2))   #QU FG
FG_280 = hp.read_map('/sharefs/alicpt/users/chenwz/Testarea/FORECAST/Foreground/foreground_Simons/gaussian_toy_fg_280GHz.fits',field=(1,2))   #QU FG
QU_FG = {27:FG_27, 39:FG_39, 93:FG_93, 145:FG_145, 225:FG_225, 280 :FG_280}
for i,map in QU_FG.items():
    QU_FG[i] = [hp.ud_grade(map[0],1024),hp.ud_grade(map[1],1024)]
"""

FG_27_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/ALL_MAP_2048/FG/27/IQU_FG_{file_indices}.fits"
FG_39_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/ALL_MAP_2048/FG/39/IQU_FG_{file_indices}.fits"
FG_93_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/ALL_MAP_2048/FG/93/IQU_FG_{file_indices}.fits"
FG_145_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/ALL_MAP_2048/FG/145/IQU_FG_{file_indices}.fits"
FG_225_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/ALL_MAP_2048/FG/225/IQU_FG_{file_indices}.fits"
FG_280_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/ALL_MAP_2048/FG/280/IQU_FG_{file_indices}.fits"




def post_cal():
    print('setting parameters...')
    nside = 1024   
    lmax = 3*nside-1 
    lmax_len = lmax # desired lmax of the lensed field.
    dlmax = lmax    # lmax of the unlensed fields is lmax + dlmax.  (some buffer is required for accurate lensing at lmax)
    epsilon = 1e-6  # target accuracy of the output maps (execution time has a fairly weak dependence on this)
    lmax_unl, mmax_unl = lmax_len + dlmax, lmax_len + dlmax
    Tcmb = 2.7255e6
    npix = hp.nside2npix(nside)

    def add_beam(fwhm_ac,nside,maps,lmax=None,pixwin=False):
        if lmax == None:
            lmax =3*nside-1
        transf = bl(fwhm_ac, nside=nside, lmax=lmax, pixwin=pixwin)
        tlm_len = hp.map2alm(maps[0], lmax=lmax)
        elm_len, blm_len = hp.map2alm_spin([maps[1], maps[2]], 2, lmax=lmax)
        Tlen = hp.alm2map(hp.almxfl(tlm_len, transf, inplace=True), nside, verbose=False)
        Qlen, Ulen = hp.alm2map_spin([hp.almxfl(elm_len, transf), hp.almxfl(blm_len, transf)], nside, 2, lmax)
        return Tlen,Qlen,Ulen

    lmaxs = {27:210, 39:210, 93:320, 145:600, 225:920, 280:1100}  #这里增大了27和39的lmax, 以超过200
    nlev = {27:25, 39:17, 93:1.9, 145:2.1, 225:4.2, 280:10}
    theta_ac = {27:91, 39:63, 93:30, 145:17, 225:11, 280:9}


    ##########################################################################
    ##########################    generate obs     ###########################
    TQU = hp.read_map(f"{CMB_2048_filename}",field=(0,1,2),)
    TQU = [replace_fwhm_map(nside,lmax,TQU[0],1.4,0,True,False),replace_fwhm_map(nside,lmax,TQU[1],1.4,0,True,False),replace_fwhm_map(nside,lmax,TQU[2],1.4,0,True,False)]
    Q = hp.ud_grade(TQU[1],nside)
    U = hp.ud_grade(TQU[2],nside)
    
    noise_dir = {27:noise_27_filename,39:noise_39_filename,93:noise_93_filename,145:noise_145_filename,225:noise_225_filename,280:noise_280_filename}
    FG_dir = {27:FG_27_filename,39:FG_39_filename,93:FG_93_filename,145:FG_145_filename,225:FG_225_filename,280:FG_280_filename}
    mask = hp.read_map(f'/sharefs/alicpt/users/chenwz/Testarea/FORECAST/mask/AliCPT_20uKcut150_C_{nside}.fits')
    mask_cut_sm = hp.read_map(f'/sharefs/alicpt/users/chenwz/Testarea/FORECAST/mask/mask{nside}_cut_sm.fits')
    mask_cut = hp.read_map(f'/sharefs/alicpt/users/chenwz/Testarea/FORECAST/mask/AliCPT_UNPfg_filled_C_{nside}.fits')

    B_obses = {}
    B_noises = {}
    B_FGs = {}
    QU_FG = {}
    for i,noise in noise_dir.items():
        TQU_sm = add_beam(theta_ac[i],  nside, [Q*mask,Q*mask,U*mask], lmax=lmaxs[i], pixwin=False)

        IQU_FG = hp.read_map(f"{FG_dir[i]}",field=(0,1,2,))
        QU_FG[i] = [hp.ud_grade(IQU_FG[1],nside),hp.ud_grade(IQU_FG[2],nside)]
        QQU_FG_sm = add_beam(theta_ac[i], nside, [QU_FG[i][0]*mask,QU_FG[i][0]*mask,QU_FG[i][1]*mask,], lmax=lmaxs[i], pixwin=False) 

        #QU_noise = hp.read_map(f"{noise}",field=(0,1,))
        QU_noise = map_compression(2048,dat=np.load(noise),flag='d2f')
        QU_noise = [hp.ud_grade(QU_noise[0],nside),hp.ud_grade(QU_noise[1],nside)]
        QQU_noise_cut = map_cut(nside,[QU_noise[0],QU_noise[0],QU_noise[1]],lmaxs[i])

        Q_all, U_all = (TQU_sm[1] + QQU_noise_cut[1]*mask + QQU_FG_sm[1]), (TQU_sm[2] + QQU_noise_cut[2]*mask + QQU_FG_sm[2])

        TQU_deconv = decon_map(theta_ac[i], nside, [Q_all, Q_all, U_all], lmaxs[i], pixwin=False)
        QQU_FG_deconv = decon_map(theta_ac[i], nside, [QQU_FG_sm[0], QQU_FG_sm[1], QQU_FG_sm[2]], lmaxs[i], pixwin=False)
        QQU_noise_deconv = decon_map(theta_ac[i], nside, [QQU_noise_cut[0], QQU_noise_cut[1], QQU_noise_cut[2]], lmaxs[i], pixwin=False)

        #_,B_deconv = qu2eb_map(nside,lmaxs[i],TQU_deconv[1],TQU_deconv[2])
        #_,B_noise_deconv = qu2eb_map(nside,lmaxs[i],QQU_noise_deconv[1],QQU_noise_deconv[2])
        #_,B_FG_deconv = qu2eb_map(nside,lmaxs[i],QQU_FG_deconv[1],QQU_FG_deconv[2])
        #B_FG_deconv,_,_ = E2Bcorrection(nside,50,[QQU_FG_deconv[0],QQU_FG_deconv[1],QQU_FG_deconv[2]],mask_bi,lmax=3000,flag='clean',n_iter=4, is_Dell=True)

        B_deconv = E2Bcorrection_new(nside,50,[TQU_deconv[1],TQU_deconv[1],TQU_deconv[2]],mask,mask_cut_sm,lmax=None,flag='clean', cal_cl=False, n_iter=4, is_Dell=True)
        B_noise_deconv = E2Bcorrection_new(nside,50,[QQU_noise_deconv[0],QQU_noise_deconv[1],QQU_noise_deconv[2]],mask,mask_cut_sm,lmax=None,flag='clean', cal_cl=False, n_iter=4, is_Dell=True)
        B_FG_deconv = E2Bcorrection_new(nside,50,[QQU_FG_deconv[0],QQU_FG_deconv[1],QQU_FG_deconv[2]],mask,mask_cut_sm,lmax=None,flag='clean', cal_cl=False, n_iter=4, is_Dell=True)
        


        B_obses[i] = B_deconv 
        B_noises[i] = B_noise_deconv
        B_FGs[i] = B_FG_deconv


    dlbb_FGs = {}
    for i,_ in B_FGs.items():
        for j,_ in B_FGs.items():
            if j<i: 
                continue
            key = (i, j)  # 可以使用元组作为字典的键
            l_bin,dlbb_FG = get_pseudo_cl(nside,mask_cut,'00',20,lmax=1000 ,apo=True,f0=[B_FGs[i],B_FGs[j]],f2=None,is_Dell=True)
            dlbb_FGs[key] = dlbb_FG
        with open(f'{FG_filename}', 'wb') as f:
            pickle.dump(dlbb_FGs, f)

    dlbb_noises = {}
    for i,_ in B_noises.items():
        for j,_ in B_noises.items():
            if j<i: 
                continue
            key = (i, j)  # 可以使用元组作为字典的键
            l_bin,dlbb_noise = get_pseudo_cl(nside,mask_cut,'00',20,lmax=1000 ,apo=True,f0=[B_noises[i],B_noises[j]],f2=None,is_Dell=True)
            dlbb_noises[key] = dlbb_noise
    with open(f'{noise_FG_filename}', 'wb') as f:
        pickle.dump(dlbb_noises, f)

    
    A = hp.read_map(f"{temp_filename}",field=(0,1,2,3,4,5,6,7,8,9,10,))    #失误，LT存成了nside=2048的map
    B_load = [A[0],A[1],A[2],A[3],A[4],A[5],A[6],A[7],A[8],A[9],A[10]]
    #for i,map in enumerate(B_load):
        #B_load[i] = hp.ud_grade(map,nside)                      #Save to nside=nside, for the convenience of storage and further analysis
    Bmap_noisy,Bmap_noise,Bmap_pure,Bmap_noisy_origin,Bmap_noise_origin,Blen,Bmap_noise_fg,Bmap_noisy_de,Bmap_pure_de,Bmap_fg_origin,Bmap_noise_differ = B_load[0],B_load[1],B_load[2],B_load[3],B_load[4],B_load[5],B_load[6],B_load[7],B_load[8],B_load[9],B_load[10]
    
    ######################## For template auto-power spectrum ################
    l_bin,dlbb_temp_noisy = get_pseudo_cl(nside,mask_cut,'00',20,lmax=1000 ,apo=True,f0=[Bmap_noisy,Bmap_noisy],f2=None,is_Dell=True)
    l_bin,dlbb_temp_noise = get_pseudo_cl(nside,mask_cut,'00',20,lmax=1000 ,apo=True,f0=[Bmap_noise,Bmap_noise],f2=None,is_Dell=True)
    l_bin,dlbb_pure = get_pseudo_cl(nside,mask_cut,'00',20,lmax=1000 ,apo=True,f0=[Bmap_pure,Bmap_pure],f2=None,is_Dell=True)
    l_bin,dlbb_temp_cross_bias = get_pseudo_cl(nside,mask_cut,'00',20,lmax=1000 ,apo=True,f0=[Bmap_noise+Bmap_noise_fg,Bmap_pure],f2=None,is_Dell=True)   #NEW CROSS BIAS DUE TO PHI NOISE
    #################cross term is negligible, so we do not calculate it.################

    ######################## For template auto-power spectrum transfer function ########################
    l = np.arange(lmax+1)
    clbb_len = hp.anafast([Q,Q,U])[2]
    dlbb_len = l*(l+1)*clbb_len[:lmax+1]/2/np.pi 
    _,dlbb_len_bin = get_binned(nside,20,dlbb_len[:1000],999)
    transfer_temp = dlbb_pure[0,:49]*cli(dlbb_len_bin)
    np.savetxt(f"{LT_LT_filename}",[l_bin, dlbb_temp_noisy[0,:49], dlbb_temp_noise[0,:49], dlbb_pure[0,:49], dlbb_temp_cross_bias[0,:49], dlbb_len_bin[:49], transfer_temp ])

    #First, all frequency maps correlate with LT map, also bias terms and transfer function are calculated
    dlbb_obs_LT = {}
    for i,B_obs in B_obses.items():
        l_bin,dlbb_cross_pure = get_pseudo_cl(nside,mask_cut,'00',20,lmax=1000 ,apo=True,f0=[Bmap_pure,B_obses[i]],f2=None,is_Dell=True)
        l_bin,dlbb_cross = get_pseudo_cl(nside, mask_cut, '00', 20, lmax=1000, apo=True,f0=[Bmap_noisy,B_obses[i]],f2=None,is_Dell=True)
        l_bin,dlbb_cross_noise1 = get_pseudo_cl(nside,mask_cut,'00',20,lmax=1000 ,apo=True,f0=[B_noises[i],Bmap_noise],f2=None,is_Dell=True)
        l_bin,dlbb_cross_noise2 = get_pseudo_cl(nside,mask_cut,'00',20,lmax=1000 ,apo=True,f0=[Blen,Bmap_noise],f2=None,is_Dell=True)
        transfer_temp_cross = dlbb_cross_pure[0,:49]*cli(dlbb_len_bin)
        l_bin,dlbb_cross_fg_temp = get_pseudo_cl(nside,mask_cut,'00',20,lmax=1000 ,apo=True,f0=[B_FGs[i],Bmap_noisy],f2=None,is_Dell=True)     #lensing template from fg
        dlbb_obs_LT[i] = [dlbb_cross_pure[0,:49], dlbb_cross[0,:49],  dlbb_cross_noise1[0,:49], dlbb_cross_noise2[0,:49], transfer_temp_cross, dlbb_cross_fg_temp[0,:49]]
    with open(f'{obs_LT_filename}', 'wb') as f:
        pickle.dump(dlbb_obs_LT, f)

    #Second, frequency maps correlate with each other 
    dlbb_obs_obs = {}
    for i,B_obs in B_obses.items():
        for j,B_obs in B_obses.items():
            if j<i: 
                continue
            l_bin,dlbb_cross = get_pseudo_cl(nside,mask_cut,'00',20,lmax=1000 ,apo=True,f0=[B_obses[i],B_obses[j]],f2=None,is_Dell=True)
            key = (i, j)  # 可以使用元组作为字典的键
            dlbb_obs_obs[key] = dlbb_cross[0,:49]
    with open(f'{obs_obs_filename}', 'wb') as f:
        pickle.dump(dlbb_obs_obs, f)
    
    return None

if __name__ == "__main__":

    # 调用 delensing_ones 函数
    post_cal()
 
            