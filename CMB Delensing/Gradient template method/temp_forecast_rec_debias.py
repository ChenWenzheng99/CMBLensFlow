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

def replace_fwhm_alm(alm,fwhm_old,fwhm_new,pix_old,pix_new,lmax):
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
    alm_deconv = replace_fwhm_alm(alm,fwhm_old,fwhm_new,pix_old,pix_new,lmax)
    map_deconv = hp.alm2map(alm_deconv,nside)
    return map_deconv

def get_pseudo_cl(nside,mask_input,flag,bin,lmax=None,apo=False,f0=None,f2=None,is_Dell=True):
    """
        nside:
        mask_input:最好是1024的mask,否则apodization很慢;或者输入apodization后的mask
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

##########################################################################
##########################    Load map dir     ###########################

# 获取传递给脚本的参数
args = sys.argv[1:]

# 打印当前作业的编号
# 计算当前作业的文件编号列表
file_indices = comm_rank + int(args[0])

# 打印当前作业的文件编号列表
print(f"Job {comm_rank} will process files: {file_indices}")


# map读取路径 /map_TQU_2048_0000.fits
TQU_filename = f"/sharefs/alicpt/users/chenwz/reconstruction_2048_simons/test/ALILENS/sims/cmbs/map_TQU_2048_{file_indices:04d}.fits"
P_filename = f"/sharefs/alicpt/users/chenwz/reconstruction_2048_simons/test/ALILENS/sims/cmbs/map_P_2048_{file_indices:04d}.fits"
rec_qlm_filename = f"/sharefs/alicpt/users/chenwz/reconstruction_multi_process2/process_added/com_rec/rec_qlm_{file_indices}.fits"

wiener_dat = np.loadtxt('/sharefs/alicpt/users/chenwz/reconstruction_multi_process2/process_added/nlpp_com.dat')
wiener = (wiener_dat[2] - wiener_dat[1]) * cli(wiener_dat[2])

QUP_N_LAT_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/ALL_MAP_2048/QUP_NOISE/LAT/93/QUP_N_{file_indices}.npy"
QUP_N_SAT_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/ALL_MAP_2048/QUP_NOISE/SAT/93/QUP_N_{file_indices}.npy"
TQU_FG_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/ALL_MAP_2048/FG/93/IQU_FG_{file_indices}.fits"

output_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/FOREGROUND_FORECAST/output_temp_rec_debias/TEMP_MAP_{file_indices}.fits"

print(f"Job {comm_rank} will read map from {TQU_filename }:")
print(f"Job {comm_rank} will write map to {output_filename }:")

##########################################################################

def delensing_ones(nside,nlev,theta_ac,):
    print('setting parameters...')
    nside = nside
    lmax = 3*nside-1 
    lmax_len = lmax # desired lmax of the lensed field.
    dlmax = lmax    # lmax of the unlensed fields is lmax + dlmax.  (some buffer is required for accurate lensing at lmax)
    epsilon = 1e-6  # target accuracy of the output maps (execution time has a fairly weak dependence on this)
    lmax_unl, mmax_unl = lmax_len + dlmax, lmax_len + dlmax
    Tcmb = 2.7255e6
    npix = hp.nside2npix(nside)


    TQU = hp.read_map(f"{TQU_filename}",field=(0,1,2),)
    TQU = [replace_fwhm_map(nside,lmax,TQU[0],1.4,0,True,False),replace_fwhm_map(nside,lmax,TQU[1],1.4,0,True,False),replace_fwhm_map(nside,lmax,TQU[2],1.4,0,True,False)]
    P = hp.read_map(f"{P_filename}",field=(0),)
    #rec_qlm = hp.read_alm(f"{rec_qlm_filename}")

    QUP_N_LAT = map_compression(nside,dat=np.load(QUP_N_LAT_filename),flag='d2f')   #注意：修改了noise 存储格式，现在只有QU noise.
    QUP_N_SAT = map_compression(nside,dat=np.load(QUP_N_SAT_filename),flag='d2f')
    #QUP_N_LAT = hp.read_map(f"{QUP_N_LAT_filename}",field=(0,1,))   #注意：修改了noise 存储格式，现在只有QU noise.
    #QUP_N_SAT = hp.read_map(f"{QUP_N_SAT_filename}",field=(0,1,))
    ##########################################################################
    ##########################  generate noise     ###########################
    ############  "1" represents LAT , "2" represents SAT  ###################

    nlev = nlev
    theta_ac = theta_ac
    ac2rad = np.pi/10800.
    theta_rad = np.array(theta_ac)*ac2rad


    fg1 = hp.read_map(f'{TQU_FG_filename}',field=[0,1,2])
    fg2 = hp.read_map(f'{TQU_FG_filename}',field=[0,1,2])

    noise1 = np.array([QUP_N_LAT[0],QUP_N_LAT[0],QUP_N_LAT[1]])
    noise2 = np.array([QUP_N_SAT[0],QUP_N_SAT[0],QUP_N_SAT[1]])


    def add_beam(fwhm_ac,nside,maps,lmax=None,pixwin=False):
        if lmax == None:
            lmax =3*nside-1
        transf = bl(fwhm_ac, nside=nside, lmax=lmax, pixwin=pixwin)
        tlm_len = hp.map2alm(maps[0], lmax=lmax)
        elm_len, blm_len = hp.map2alm_spin([maps[1], maps[2]], 2, lmax=lmax)
        Tlen = hp.alm2map(hp.almxfl(tlm_len, transf, inplace=True), nside, verbose=False)
        Qlen, Ulen = hp.alm2map_spin([hp.almxfl(elm_len, transf), hp.almxfl(blm_len, transf)], nside, 2, lmax)
        return Tlen,Qlen,Ulen


    ###########################################################################
    ##########################     prepare maps     ###########################
    Qlen_full, Ulen_full, Pmap_full = TQU[1], TQU[2], P

    nlpp = wiener_dat[1]  
    phi_noise = hp.synfast(nlpp, nside, lmax=3*nside-1)
    hp.write_map(f'/sharefs/alicpt/users/chenwz/Testarea/FORECAST/ALL_MAP_2048/QUP_NOISE/Simons_93_145_P_noise/Gaussian_phi_noise_{file_indices}.fits',phi_noise,overwrite=True)

    ######################## Cut the fake phi map (to be consistent with the real phi map) ##############################
    cut = np.where((np.arange(dlmax + 1) > 10) * (np.arange(dlmax + 1) < 3072), 1, 0) # band limit
    Pmap_full = hp.alm2map(hp.almxfl(hp.map2alm(Pmap_full), cut),nside)   #cut reconstructed phi map
    phi_noise = hp.alm2map(hp.almxfl(hp.map2alm(phi_noise), cut),nside)   #cut reconstructed phi map
    #####################################################################################################################

    #Prec = hp.alm2map(hp.almxfl(rec_qlm, cut),nside)   #cut reconstructed phi map

     
    mask = hp.read_map(f'/sharefs/alicpt/users/chenwz/Testarea/FORECAST/mask/mask_{nside}_Sm.fits')
    mask_sm = hp.read_map(f'/sharefs/alicpt/users/chenwz/Testarea/FORECAST/mask/mask_{nside}_Sm.fits')   #smoothed mask is necessary

    #mask_cut = hp.read_map(f'/sharefs/alicpt/users/chenwz/Testarea/FORECAST/mask/AliCPT_UNPfg_filled_C_{nside}.fits')
    mask_cut_sm = hp.read_map(f'/sharefs/alicpt/users/chenwz/Testarea/FORECAST/mask/mask{nside}_cut_sm.fits')

    #mask = np.ones_like(mask)

    mask = mask_sm

    mask1 = mask_sm   #fsky~0.1
    mask2 = mask_sm    #fsky~0.1

    fg_obs1 = add_beam(theta_ac[0],nside,fg1,lmax=None,pixwin=False)
    fg_obs2 = add_beam(theta_ac[1],nside,fg2,lmax=None,pixwin=False)

    _,Qlen_full1,Ulen_full1 = add_beam(theta_ac[0],nside,[Qlen_full,Qlen_full,Ulen_full],lmax=None,pixwin=False)
    _,Qlen_full2,Ulen_full2 = add_beam(theta_ac[1],nside,[Qlen_full,Qlen_full,Ulen_full],lmax=None,pixwin=False)


    Qlen_part1,Ulen_part1 = Qlen_full1*mask1, Ulen_full1*mask1
    Qlen_part2,Ulen_part2 = Qlen_full2*mask2, Ulen_full2*mask2

    noise_part1 = (noise1 + fg_obs1)*mask1                                            #noise QU1
    noise_part2 = (noise2 + fg_obs2)*mask2                                            #noise QU2

    #combine maps


    Qlen_noisy_part1, Ulen_noisy_part1 = Qlen_part1 + noise_part1[1], Ulen_part1 + noise_part1[2]   #noisy QU1
    Qlen_noisy_part2, Ulen_noisy_part2 = Qlen_part2 + noise_part2[1], Ulen_part2 + noise_part2[2]   #noisy QU2

    phi_noisy_part = (Pmap_full + phi_noise)*mask2                                       #noisy phi

    Pmap_part = Pmap_full*mask2                                           #signal phi

    phi_noise_part = phi_noise*mask2                                       #noise phi

    ###########################################################################

    def decon_alm(alm,fwhm,lmax):
        """
        deconvolve alm with beam
        """
        bl = hp.gauss_beam(fwhm,lmax=lmax,pol=True)
        alm_deconv = hp.almxfl(alm,1/bl[:,1])
        return alm_deconv

    def decon_map(nside,lmax,map,fwhm):
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

    ###########################################################################

    def map2alm_stack(lmax, *maps, fwhm=[0]):
        """
        convert maps to alm, and stack them together
        Warning: This function perform beam deconvolution, if you dont want to deconvolve the beam, please set fwhm=0
        """
        qlms = []
        ulms = []
        

        # 对每组 map 进行循环
        for i in range(len(maps)):
            bl = hp.gauss_beam(fwhm=fwhm[i], lmax=lmax, pol=True)
            # 分别对 Q 和 U 分量进行 map2alm
            qlm = hp.map2alm(maps[i][0], lmax=lmax, pol=True)
            ulm = hp.map2alm(maps[i][1], lmax=lmax, pol=True)

            qlm_sm = hp.almxfl(qlm, 1/bl[:,1])
            ulm_sm = hp.almxfl(ulm, 1/bl[:,2])

            # 将 qlm 和 ulm 存入对应的列表
            qlms.append(qlm_sm)
            ulms.append(ulm_sm)

        # 使用 np.column_stack 将 qlms 和 ulms 组合
        stacked_qlms = np.column_stack(qlms)
        stacked_ulms = np.column_stack(ulms)

        return stacked_qlms, stacked_ulms

    def map2cl_stack(lmax, *maps, fwhm=[0]):
        """
        convert maps to cl, and stack them together
        Warning: This function perform beam deconvolution, if you dont want to deconvolve the beam, please set fwhm=0
        """
        power = []
        
        for i in range(len(maps)):
            bl = hp.gauss_beam(fwhm=fwhm[i], lmax=lmax, pol=True)
            alm = hp.map2alm_spin(maps[i], 2, lmax=lmax, mmax=lmax)
            elm_sm = hp.almxfl(alm[0], 1/bl[:,1])
            power.append(hp.alm2cl(elm_sm, elm_sm, lmax, ))
        stacked_power = np.column_stack(power)
        return stacked_power

    ###########################################################################
    ###########       prepare for inverse variance combining       ############
    #####  Automatically deconvolved, which is necessary when combining  ######

    #stacked noisy lensed QU alm
    qlm_noisy_stacked,ulm_noisy_stacked = map2alm_stack(lmax, [Qlen_noisy_part1, Ulen_noisy_part1],[Qlen_noisy_part2, Ulen_noisy_part2], fwhm=theta_rad)   #lensed CMB + noise + fg

    #stacked QU noise alm
    qlm_noise_stacked,ulm_noise_stacked = map2alm_stack(lmax, [noise_part1[1], noise_part1[2]],[noise_part2[1], noise_part2[2]], fwhm=theta_rad)   #noise + fg

    #FOR SPECIAL USAGE: NOISE WITHOUT FOREGROUND
    qlm_noisy_no_fg_stacked,ulm_noisy_no_fg_stacked = map2alm_stack(lmax, [Qlen_noisy_part1-fg_obs1[1]*mask1, Ulen_noisy_part1-fg_obs1[2]*mask1],[Qlen_noisy_part2-fg_obs2[1]*mask2, Ulen_noisy_part2-fg_obs2[2]*mask2], fwhm=theta_rad)   #lensed CMB + noise

    qlm_noise_no_fg_stacked,ulm_noise_no_fg_stacked = map2alm_stack(lmax, [noise1[1]*mask1, noise1[2]*mask1],[noise2[1]*mask2, noise2[2]*mask2], fwhm=theta_rad)   #noise

    #FOR SPECIAL USAGE:  FOREGROUND
    qlm_fg_stacked,ulm_fg_stacked = map2alm_stack(lmax, [fg_obs1[1]*mask1, fg_obs1[2]*mask1],[fg_obs2[1]*mask2, fg_obs2[2]*mask2], fwhm=theta_rad)   #fg

    #stacked pure lensed QU alm
    qlm_pure_stacked, ulm_pure_stacked = map2alm_stack(lmax, [Qlen_part1, Ulen_part1],[Qlen_part2, Ulen_part2], fwhm=theta_rad)   #lensed CMB

    #noisy phi alm
    plm_noisy = hp.map2alm(phi_noisy_part,dlmax,)

    #phi noise alm
    plm_noise = hp.map2alm(phi_noise_part,dlmax,)

    #pure signal phi alm
    plm_pure = hp.map2alm(Pmap_full*mask2)

    #stacked EE noise
    ee_noise_stacked = map2cl_stack(lmax, [noise_part1[1], noise_part1[2]],[noise_part2[1], noise_part2[2]], fwhm=theta_rad)

    def combine_with_lcut_new(ee_noise, qlm, ulm, Lmin, Lmax):
        """
        Input several sets of alm and Nlee, combine them with inverse variance weighting for public mutlipoles, and remain intact for each own private multipoles, set alm to zero for those out of range.
        For example, there are two alm sets, one is [lmin1,lmax1], another is [lmin2,lmax2], then the combined alm will be:

        ------+++++********----------
        -----------********++++------

        where ------ are set to zero, ******** are combined with inverse variance weighting, and +++++ are kept intact.

        """
        Qlm = np.zeros(np.shape(qlm)[0],dtype=complex)
        Ulm = np.zeros(np.shape(ulm)[0],dtype=complex)

        ee_noise_modified = ee_noise.copy()
        for i, (lmin, lmax) in enumerate(zip(Lmin, Lmax)):
            ee_noise_modified[:lmin,i] = 1e60
            ee_noise_modified[lmax+1:,i] = 1e60
        ee_noise_inv = cli(ee_noise_modified)

        llow = np.min(Lmin)
        ltop = np.max(Lmax)

        weights = []    #先得公共区域weight=inverse_noise_square
        for i, (lmin, lmax) in enumerate(zip(Lmin, Lmax)):
            weight = ee_noise_inv[:,i] * cli(np.sum(ee_noise_inv, axis=1))
            weights.append(weight)
        weights = np.array(weights).T

        weights[:llow+1] = 0
        weights[ltop:] = 0

        for i, (lmin, lmax) in enumerate(zip(Lmin, Lmax)):
            Qlm += (hp.almxfl(qlm[:,i] , weights[:,i]) ) #almxfl：注意：若weight的lmax小于alm的lmax，则weight会被自动补零,这会导致后面的alm变成0
            Ulm += (hp.almxfl(ulm[:,i] , weights[:,i]) )

        return Qlm,Ulm
    
    Lmin=[200,20]
    Lmax=[4800,320]

    qlm_noisy,ulm_noisy = combine_with_lcut_new(ee_noise_stacked, qlm_noisy_stacked, ulm_noisy_stacked, Lmin, Lmax)      #combined noisy QU alm
    qlm_noise,ulm_noise = combine_with_lcut_new(ee_noise_stacked, qlm_noise_stacked, ulm_noise_stacked, Lmin, Lmax)      #combined QU noise alm
    qlm_pure,ulm_pure = combine_with_lcut_new(ee_noise_stacked, qlm_pure_stacked, ulm_pure_stacked, Lmin, Lmax)          #combined pure QU alm

    Qlen_noisy_part = hp.alm2map(qlm_noisy, nside, lmax, lmax, )
    Ulen_noisy_part = hp.alm2map(ulm_noisy, nside, lmax, lmax, )

    Qnoise_part = hp.alm2map(qlm_noise, nside, lmax, lmax, )
    Unoise_part = hp.alm2map(ulm_noise, nside, lmax, lmax, )

    #FOR SPECIAL USAGE: NOISE WITHOUT FOREGROUND
    qlm_noisy_no_fg,ulm_noisy_no_fg = combine_with_lcut_new(ee_noise_stacked, qlm_noisy_no_fg_stacked, ulm_noisy_no_fg_stacked, Lmin, Lmax)      #combined noisy QU alm
    qlm_noise_no_fg,ulm_noise_no_fg = combine_with_lcut_new(ee_noise_stacked, qlm_noise_no_fg_stacked, ulm_noise_no_fg_stacked, Lmin, Lmax)      #combined QU noise alm

    Qlen_noisy_no_fg_part = hp.alm2map(qlm_noisy_no_fg, nside, lmax, lmax, )
    Ulen_noisy_no_fg_part = hp.alm2map(ulm_noisy_no_fg, nside, lmax, lmax, )

    Qlen_noise_no_fg_part = hp.alm2map(qlm_noise_no_fg, nside, lmax, lmax, )
    Ulen_noise_no_fg_part = hp.alm2map(ulm_noise_no_fg, nside, lmax, lmax, )

    #FOR SPECIAL USAGE: FOREGROUND
    qlm_fg,uln_fg = combine_with_lcut_new(ee_noise_stacked, qlm_fg_stacked, ulm_fg_stacked, Lmin, Lmax)      #combined fg

    Qlen_fg_part = hp.alm2map(qlm_fg, nside, lmax, lmax, )
    Ulen_fg_part = hp.alm2map(uln_fg, nside, lmax, lmax, )

    ###########################################################################
    ###############     get deconvolved 'input' noisy maps     ################   
    Bmap_noisy_origin,_,_ = E2Bcorrection(nside,50,[Qlen_noisy_part,Qlen_noisy_part,Ulen_noisy_part],mask_sm,lmax=3000,flag='clean',n_iter=4, is_Dell=True)

    Bmap_noise_origin,_,_ = E2Bcorrection(nside,50,[Qlen_noise_no_fg_part,Qlen_noise_no_fg_part,Ulen_noise_no_fg_part],mask_sm,lmax=3000,flag='clean',n_iter=4, is_Dell=True)

    Bmap_fg_origin,_,_ = E2Bcorrection(nside,50,[Qlen_fg_part,Qlen_fg_part,Ulen_fg_part],mask_sm,lmax=3000,flag='clean',n_iter=4, is_Dell=True)

    ###########################################################################
    ########################        Wiener_filter     #########################
    def Wiener_filter_phi_new(nside,lmax,pure_map,noisy_map,):
        plm_pure = curvedsky.utils.hp_map2alm(nside, lmax, lmax, pure_map)
        plm_noisy_full = curvedsky.utils.hp_map2alm(nside, lmax, lmax, noisy_map)
        clpp_pure_full = curvedsky.utils.alm2cl(lmax, plm_pure, alm2=None)
        clpp_noisy_full = curvedsky.utils.alm2cl(lmax, plm_noisy_full, alm2=None)
        ratio = (clpp_pure_full) * cli(clpp_noisy_full)
        return ratio

    ratio1 = Wiener_filter_QU(nside,lmax,[Qlen_noisy_part, Qlen_noisy_part, Ulen_noisy_part],[Qnoise_part,Qnoise_part,Unoise_part],)
    #ratio2 = Wiener_filter_phi(nside,dlmax,Pmap_part,phi_noisy_part)
    ratio2 = wiener

    Qlm_noisy = hp.almxfl(qlm_noisy, ratio1)
    Ulm_noisy = hp.almxfl(ulm_noisy, ratio1)

    Qlm_noise = hp.almxfl(qlm_noise, ratio1)
    Ulm_noise = hp.almxfl(ulm_noise, ratio1)

    Qlm_pure = hp.almxfl(qlm_pure, ratio1)
    Ulm_pure = hp.almxfl(ulm_pure, ratio1)

    Qlm_noisy_no_fg = hp.almxfl(qlm_noisy_no_fg, ratio1)
    Ulm_noisy_no_fg = hp.almxfl(ulm_noisy_no_fg, ratio1)

    Qlm_noise_only = hp.almxfl(qlm_noise_no_fg, ratio1)
    Ulm_noise_only = hp.almxfl(ulm_noise_no_fg, ratio1)

    Plm_noisy = hp.almxfl(plm_noisy, ratio2)

    Plm_noise = hp.almxfl(plm_noise, ratio2)

    Plm_pure = hp.almxfl(plm_pure, ratio2)

    elm_noisy,blm_noisy = qu2eb_alm(nside,lmax,Qlm_noisy,Ulm_noisy)    #这里忽略了B2E的泄露，因为对Emode来说影响很小
    elm_noise,blm_noise = qu2eb_alm(nside,lmax,Qlm_noise,Ulm_noise)
    elm_pure,blm_pure = qu2eb_alm(nside,lmax,Qlm_pure,Ulm_pure)
    elm_noise_only,blm_noise_only = qu2eb_alm(nside,lmax,Qlm_noise_only,Ulm_noise_only)
    elm_noisy_no_fg,blm_noise_no_fg = qu2eb_alm(nside,lmax,Qlm_noisy_no_fg,Ulm_noisy_no_fg)

    Plm_noisy = curvedsky.utils.lm_healpy2healpix(Plm_noisy, lmax, lmpy=0)
    Plm_noise = curvedsky.utils.lm_healpy2healpix(Plm_noise, lmax, lmpy=0)
    Plm_pure = curvedsky.utils.lm_healpy2healpix(Plm_pure, lmax, lmpy=0)

    ###########################################################################
    ##################       calculate lensing template     ###################

    elmax=4800          ################注意：这里要手动设定。
    dlmax=4800          ################注意：这里要手动设定。其实取lmax也可以，因为wiener filter起到了截断low SNR的multipole的作用。
    elmin=20   
    dlmin=10   

    #注意：之前的处理需要smoothed mask,不要使用binary mask
    Bmap_noisy,_ = lensingb_mine(lmax, elmin, elmax, dlmin, dlmax, elm_noisy, Plm_noisy, nside_t=nside, gtype='p',sm_mask=mask)
    Bmap_noise1,_ = lensingb_mine(lmax, elmin, elmax, dlmin, dlmax, elm_noise_only, Plm_pure, nside_t=nside, gtype='p',sm_mask=mask)
    Bmap_noise2,_ = lensingb_mine(lmax, elmin, elmax, dlmin, dlmax, elm_noisy_no_fg, Plm_noise, nside_t=nside, gtype='p',sm_mask=mask)
    Bmap_pure,map_pure = lensingb_mine(lmax, elmin, elmax, dlmin, dlmax, elm_pure, Plm_pure, nside_t=nside, gtype='p',sm_mask=mask)
    Bmap_noise_fg,_ = lensingb_mine(lmax, elmin, elmax, dlmin, dlmax, elm_noisy-elm_noisy_no_fg, Plm_noisy, nside_t=nside, gtype='p',sm_mask=mask)

    ###########################################################################
    ##################       noise & signal post process     ##################

    Bmap_noise =  (Bmap_noise1 + Bmap_noise2)
    Bmap_noise_differ = (Bmap_noise_origin - Bmap_noise)
    Bmap_noisy_de = (Bmap_noisy_origin - Bmap_noisy)

    _,Bmap_full = qu2eb_map(nside,lmax,Qlen_full,Ulen_full)
    Bmap_pure_de = (Bmap_full - Bmap_pure)
    Elen,Blen = qu2eb_map(nside,lmax,Qlen_full,Ulen_full)

    ###########################################################################
    ##################       calculate pseudo cl     ##########################    !!!!!!!!!!!!This is for self-check, rather than further analysis.!!!!!!!!!!!!
    # Notice: the 'get_pseudo_cl' function will multiply the mask again, so do not multiply the smoothed mask before inputting.



    ######################## For template auto-power spectrum ################
    l_bin,dlbb_temp_noisy = get_pseudo_cl(nside,mask_cut_sm,'00',20,lmax=2000 ,apo=True,f0=[Bmap_noisy,Bmap_noisy],f2=None,is_Dell=True)
    l_bin,dlbb_temp_noise = get_pseudo_cl(nside,mask_cut_sm,'00',20,lmax=2000 ,apo=True,f0=[Bmap_noise,Bmap_noise],f2=None,is_Dell=True)
    l_bin,dlbb_pure = get_pseudo_cl(nside,mask_cut_sm,'00',20,lmax=2000 ,apo=True,f0=[Bmap_pure,Bmap_pure],f2=None,is_Dell=True)

    l_bin,dlbb_temp_cross_bias = get_pseudo_cl(nside,mask_cut_sm,'00',20,lmax=2000 ,apo=True,f0=[Bmap_noise+Bmap_noise_fg,Bmap_pure],f2=None,is_Dell=True)   #NEW CROSS BIAS DUE TO PHI NOISE
    #################cross term is negligible, so we do not calculate it.################

    ######################## For template cross-power spectrum ################
    l_bin,dlbb_cross_pure = get_pseudo_cl(nside,mask_cut_sm,'00',20,lmax=2000 ,apo=True,f0=[Bmap_pure,Bmap_noisy_origin],f2=None,is_Dell=True)
    l_bin,dlbb_cross = get_pseudo_cl(nside,mask_cut_sm,'00',20,lmax=2000 ,apo=True,f0=[Bmap_noisy,Bmap_noisy_origin],f2=None,is_Dell=True)

    l_bin,dlbb_cross_noise1 = get_pseudo_cl(nside,mask_cut_sm,'00',20,lmax=2000 ,apo=True,f0=[Bmap_noise_origin,Bmap_noisy],f2=None,is_Dell=True)
    l_bin,dlbb_cross_noise2 = get_pseudo_cl(nside,mask_cut_sm,'00',20,lmax=2000 ,apo=True,f0=[Blen,Bmap_noise],f2=None,is_Dell=True)

    l_bin,dlbb_cross_temp_fg = get_pseudo_cl(nside,mask_cut_sm,'00',20,lmax=2000 ,apo=True,f0=[Bmap_noisy,Bmap_fg_origin],f2=None,is_Dell=True)     #lensing template from fg

    l_bin,dlbb_temp_fg = get_pseudo_cl(nside,mask_cut_sm,'00',20,lmax=2000 ,apo=True,f0=[Bmap_noise_fg,Bmap_noise_fg],f2=None,is_Dell=True)     #lensing template from fg

    ######################## For delensed spectrum ################
    l_bin,dlbb_noisy_de = get_pseudo_cl(nside,mask_cut_sm,'00',20,lmax=2000 ,apo=True,f0=[Bmap_noisy_de,Bmap_noisy_de],f2=None,is_Dell=True)
    l_bin,dlbb_res = get_pseudo_cl(nside,mask_cut_sm,'00',20,lmax=2000 ,apo=True,f0=[Bmap_pure_de,Bmap_pure_de],f2=None,is_Dell=True)
    l_bin,dlbb_cross_res_noise = get_pseudo_cl(nside,mask_cut_sm,'00',20,lmax=2000 ,apo=True,f0=[Bmap_pure_de,Bmap_noise_origin],f2=None,is_Dell=True)
    l_bin,dlbb_cross_res_fg = get_pseudo_cl(nside,mask_cut_sm,'00',20,lmax=2000 ,apo=True,f0=[Bmap_pure_de,Bmap_fg_origin],f2=None,is_Dell=True)
    l_bin,dlbb_cross_res_tempn = get_pseudo_cl(nside,mask_cut_sm,'00',20,lmax=2000 ,apo=True,f0=[Bmap_pure_de,Bmap_noise],f2=None,is_Dell=True)
    l_bin,dlbb_cross_fg_noise = get_pseudo_cl(nside,mask_cut_sm,'00',20,lmax=2000 ,apo=True,f0=[Bmap_noise_origin,Bmap_fg_origin],f2=None,is_Dell=True)
    l_bin,dlbb_cross_noise_tempn = get_pseudo_cl(nside,mask_cut_sm,'00',20,lmax=2000 ,apo=True,f0=[Bmap_noise,Bmap_noise_origin],f2=None,is_Dell=True)
    l_bin,dlbb_cross_fg_tempn = get_pseudo_cl(nside,mask_cut_sm,'00',20,lmax=2000 ,apo=True,f0=[Bmap_noise,Bmap_fg_origin],f2=None,is_Dell=True)
    l_bin,dlbb_noise = get_pseudo_cl(nside,mask_cut_sm,'00',20,lmax=2000 ,apo=True,f0=[Bmap_noise_origin,Bmap_noise_origin],f2=None,is_Dell=True)
    l_bin,dlbb_fg = get_pseudo_cl(nside,mask_cut_sm,'00',20,lmax=2000 ,apo=True,f0=[Bmap_fg_origin,Bmap_fg_origin],f2=None,is_Dell=True)
    l_bin,dlbb_tempn = get_pseudo_cl(nside,mask_cut_sm,'00',20,lmax=2000 ,apo=True,f0=[Bmap_noise,Bmap_noise],f2=None,is_Dell=True)
    l_bin,dlbb_noise_differ = get_pseudo_cl(nside,mask_cut_sm,'00',20,lmax=2000 ,apo=True,f0=[Bmap_noise_differ,Bmap_noise_differ],f2=None,is_Dell=True)
    
    ###########################################################################
    ##############################Transfer function############################
    l = np.arange(lmax+1)
    clbb_len = hp.anafast([Qlen_full,Qlen_full,Ulen_full])[2]
    dlbb_len = l*(l+1)*clbb_len[:lmax+1]/2/np.pi 

    _,dlbb_len_bin = get_binned(nside,20,dlbb_len[:1000],999)
    transfer_temp = dlbb_pure.T[:49,0]*cli(dlbb_len_bin)
    transfer_temp_cross = dlbb_cross_pure[0,:49]*cli(dlbb_len_bin)

    #######################################################
    ###############    mission1 complete    ###############
    output = [Bmap_noisy,Bmap_noise,Bmap_pure,Bmap_noisy_origin,Bmap_noise_origin,Blen,Bmap_noise_fg,Bmap_noisy_de,Bmap_pure_de,Bmap_fg_origin,Bmap_noise_differ]
    for i,map in enumerate(output):
        output[i] = hp.ud_grade(map,1024)                      #Save to nside=1024, for the convenience of storage and further analysis

    hp.write_map(f'{output_filename}',output)   

    #######################################################
    ###############    mission complete    ################
    np.savetxt(os.path.join(savePath, 'all_in_one_%03d.dat' % file_indices), [l_bin[:49], (dlbb_temp_noisy[0,:49]), (dlbb_temp_noise[0,:49]), dlbb_pure[0,:49], dlbb_temp_cross_bias[0,:49], transfer_temp, dlbb_len_bin])

    np.savetxt(os.path.join(savePath2, 'all_in_one_%03d.dat' % file_indices), [l_bin[:49], dlbb_cross[0,:49], dlbb_cross_pure[0,:49], dlbb_cross_noise1[0,:49], dlbb_cross_noise2[0,:49],
        dlbb_cross_temp_fg[0,:49], dlbb_temp_fg[0,:49], transfer_temp_cross,])
    
    np.savetxt(os.path.join(savePath3, 'all_in_one_%03d.dat' % file_indices), [l_bin[:49], dlbb_noisy_de[0,:49], dlbb_res[0,:49], dlbb_cross_res_noise[0,:49], dlbb_cross_res_fg[0,:49],
        dlbb_cross_res_tempn[0,:49], dlbb_cross_fg_noise[0,:49], dlbb_cross_noise_tempn[0,:49], dlbb_cross_fg_tempn[0,:49], dlbb_noise[0,:49], dlbb_fg[0,:49], dlbb_tempn[0,:49], dlbb_noise_differ[0,:49]],)

    return None

if __name__ == "__main__":

    nside = 2048
    nlev = [5.8,1.9]
    theta_ac = [2.2,30]

    savePath = f'/sharefs/alicpt/users/chenwz/Testarea/FORECAST/FOREGROUND_FORECAST/self_check_temp_rec_debias/auto'
    savePath2 = f'/sharefs/alicpt/users/chenwz/Testarea/FORECAST/FOREGROUND_FORECAST/self_check_temp_rec_debias/cross'
    savePath3 = f'/sharefs/alicpt/users/chenwz/Testarea/FORECAST/FOREGROUND_FORECAST/self_check_temp_rec_debias/de'

    # 调用 delensing_ones 函数
    delensing_ones(nside,nlev,theta_ac)