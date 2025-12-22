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
        f_01 = nmt.NmtField(mask, f0_1, lmax=lmax)  
        f_02 = nmt.NmtField(mask, f0_2, lmax=lmax)  

        print('pseudo cl calculating ')
        cl_00 = nmt.compute_full_master(f_01, f_02, b)    #TT为 cl_00[0]
        return ell_arr,cl_00

    # spin-0 x spin-2
    if flag == '02':
        f_0 = nmt.NmtField(mask, f0_1, lmax=lmax)  
        f_2 = nmt.NmtField(mask, f2, spin=2, purify_e=False, purify_b=False, n_iter_mask=3, lmax=lmax)  
        assert f_0 is not None
        assert f_2 is not None
        print('pseudo cl calculating ')
        cl_02 = nmt.compute_full_master(f_0, f_2, b)
        return ell_arr,cl_02
    
    # spin-2 x spin-2
    if flag == '22':
        f_2 = nmt.NmtField(mask, f2, spin=2, purify_e=False, purify_b=False, n_iter_mask=3, lmax=lmax)  
        assert f_2 is not None
        print('pseudo cl calculating ')
        cl_22 = nmt.compute_full_master(f_2, f_2, b)      #EE,BB 分别为 cl_22[0]，cl_22[3]
        return ell_arr,cl_22

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
    alm = hp.map2alm([Qmap,Qmap,Umap],lmax)
    Emap = hp.alm2map(alm[1],nside)
    Bmap = hp.alm2map(alm[2],nside)
    return Emap,Bmap

def k2palm(data, lmax):
    ll = np.arange(0, lmax + 1)
    fac = 2 / (ll*(ll+1))
    fac[0:2] = 0
    data = hp.almxfl(data, fac)
    return data

##########################################################################
##########################    Load map dir     ###########################

# 获取传递给脚本的参数
args = sys.argv[1:]

# 打印当前作业的编号
# 计算当前作业的文件编号列表
file_indices = comm_rank + int(args[0])
#file_indices = 112

# 打印当前作业的文件编号列表
print(f"Job {comm_rank} will process files: {file_indices}")


# map读取路径 /map_TQU_2048_0000.fits
TQU_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_ILC/mmDL_maps_2048/{file_indices:05d}/lensedcmb_{file_indices:05d}.fits"
P_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_ILC/mmDL_maps_2048/{file_indices:05d}/kappa_{file_indices:05d}.fits"
#rec_qlm_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MAIN/rec/QE_rec_temperature/QE_MV_red_alm/qlm_QE_MV_{file_indices:04d}.fits"
#comb_alm_filename = f"/sharefs/alicpt/users/chenwz/reconstruction_multi_process2/process_added/LSS_tracer/COMB_KIL/comb_KIL_{file_indices}.npy"


#QUP_N_LAT_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/ALL_MAP_2048/QUP_NOISE/LAT/93/QUP_N_{file_indices}.npy"
#QUP_N_SAT_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/ALL_MAP_2048/QUP_NOISE/SAT/93/QUP_N_{file_indices}.npy"
#TQU_FG_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/ALL_MAP_2048/FG/93/IQU_FG_{file_indices}.fits"

#output_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/FOREGROUND_FORECAST/output_temp_comb/TEMP_MAP_{file_indices}.fits"

#print(f"Job {comm_rank} will read map from {TQU_filename }:")
#print(f"Job {comm_rank} will write map to {output_filename }:")


# 获取传递给脚本的参数
args = sys.argv[1:]

# 打印当前作业的编号
# 计算当前作业的文件编号列表
file_indices = comm_rank + int(args[0]) #+ 200

# 打印当前作业的文件编号列表
print(f"Job {comm_rank} will process files: {file_indices}")


# map 输出路径 
cl_auto_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MAIN/revise/delens/output_cl_red/remap_QE_ASL/auto/all_in_one_{file_indices}.dat"
cl_cross_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MAIN/revise/delens/output_cl_red/remap_QE_ASL/cross/all_in_one_{file_indices}.dat"
cl_de_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MAIN/revise/delens/output_cl_red/remap_QE_ASL/de/all_in_one_{file_indices}.dat"
#cl_terms_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK/delens/remap_method/cl/terms_check/all_in_one_{file_indices}.dat"
cl_obs_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MAIN/revise/delens/output_cl_red/remap_QE_ASL/obs/all_in_one_{file_indices}.dat"



# map 输入路径
cmb_noisy_comb_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MAIN/revise/rec/prepare/comb_delens_maps/final_obs/obs_final_QU_map_{file_indices:04d}.fits"
cmb_noise_comb_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MAIN/revise/rec/prepare/comb_delens_maps/final_res/res_final_QU_map_{file_indices:04d}.fits"
#cmb_signal_comb_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK/sim/combination/cmb_signal/comb_signal_{file_indices:04d}.fits"

# map 输入路径
phi_noisy_comb_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MAIN/revise/rec/QE_HO_temperature_COMB/qlm_QE_MV_red/qlm_QE_MV_{file_indices:04d}.fits"
#phi_signal_comb_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MAIN/sim/combination/phi_signal/comb_dmap_{file_indices:04d}.fits"
#nlpp_comb_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MAIN/sim/combination/nlpp_comb/nlpp_comb_{file_indices:04d}.dat"


print(f"Job {comm_rank} will write map to {cl_auto_filename}:")


##########################################################################

def delensing_ones(nside,):
    print('setting parameters...')
    nside = nside
    lmax = 3*nside-1 
    lmax_len = lmax # desired lmax of the lensed field.
    dlmax = 5000    # lmax of the unlensed fields is lmax + dlmax.  (some buffer is required for accurate lensing at lmax)
    epsilon = 1e-6  # target accuracy of the output maps (execution time has a fairly weak dependence on this)
    lmax_unl, mmax_unl = lmax_len + dlmax, lmax_len + dlmax
    Tcmb = 2.7255e6
    npix = hp.nside2npix(nside)

    ##########################################################################
    #######################   Load lensing potential   #######################

    q2k = lambda l: l*(l + 1) / 2 # potential -> convergence
    q2d = lambda l: (l*(l + 1)) ** 0.5 # potential -> deflection

    mask = hp.read_map(f'/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MAIN/sim/mask/ALI_LAT_add_SO_LAT_cross_PLANCK_{nside}_bi.fits')
    plm_input = hp.map2alm(hp.read_map(P_filename))
    plm_input = k2palm(plm_input, lmax)
    plm_rec = hp.read_alm(phi_noisy_comb_filename)
    #wiener_dat = np.load('/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MAIN/rec/QE_rec_temperature_SO/Wiener_QE_SO_MV.npy')
    #wiener = wiener_dat[2] 
    clpp_rec = (hp.alm2cl(plm_rec)/np.mean(mask))
    wiener = hp.alm2cl(plm_input)[:len(clpp_rec)]/clpp_rec
    wiener[np.where(wiener == np.inf)] = 0
    wiener[0] = 0
    cut = np.where((np.arange(dlmax + 1) > 10) * (np.arange(dlmax + 1) < dlmax), 1, 0) # band limit

    ######################################################################################
    #######################       process phi reconstruction       #######################

    plm_input = hp.map2alm(hp.read_map(P_filename),dlmax)
    plm_input = k2palm(plm_input, dlmax)
    pmap_input = hp.alm2map(hp.almxfl(plm_input, cut * wiener), nside,)

    pmap_rec = hp.alm2map(hp.almxfl(plm_rec, cut * wiener), nside)

    ######################################################################################
    ##############################       prepare maps       ##############################

    #TQU_map_FIL = hp.read_map(f'/root/Testarea/A_NEW_WORK_MAIN/sims_47/ALI_LAT_add_SO_LAT_add_LiteBIRD_{nside}_signal_filtered_coadd.fits',field=(0,1,2))
    TQU_map_UNFIL = hp.read_map(TQU_filename,field=(0,1,2))
    #TQU_map_UNFIL = np.array([replace_fwhm_map(nside,lmax,TQU_map_UNFIL[i],1.4,0,False,False) for i in range(3)])

    ####_,Qlen_full,Ulen_full,Pmap_full = making_maps_new(nside, nrms_f=None,fwhm_f=[0],phi_map=None,pixwin=False)
    ####_,Qlen_full,Ulen_full,Pmap_full = making_maps(nside,fwhm=None)

    #Pmap_full = hp.read_map('/root/Testarea/A_NEW_WORK_MAIN/sims_47/map_P_2048_0047.fits')
    Pmap_full = pmap_input

    mask = hp.read_map(f'/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MAIN/sim/mask/ALI_LAT_add_SO_LAT_cross_PLANCK_{nside}_sm_thin_0.1.fits')
    mask_sm = hp.read_map(f'/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MAIN/sim/mask/ALI_LAT_add_SO_LAT_cross_PLANCK_{nside}_sm_thin_0.1.fits')   #smoothed mask is necessary

    mask_cut = hp.read_map(f'/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MAIN/sim/mask/ALI_LAT_add_SO_LAT_cross_PLANCK_{nside}_sm_thin_0.2.fits')
    mask_cut_sm = hp.read_map(f'/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MAIN/sim/mask/ALI_LAT_add_SO_LAT_cross_PLANCK_{nside}_sm_thin_0.2.fits')


    #CMB_TQU_FIL = hp.read_map(f'/root/Testarea/A_NEW_WORK_MAIN/sims_47/ALI_LAT_add_SO_LAT_add_LiteBIRD_{nside}_filtered_coadd.fits',field=(0,1,2))
    CMB_TQU_UNFIL = hp.read_map(cmb_noisy_comb_filename, field=(0,1))

    #RES_TQU_FIL = hp.read_map(f'/root/Testarea/A_NEW_WORK_MAIN/sims_47/ALI_LAT_add_SO_LAT_add_LiteBIRD_{nside}_filtered_noise_coadd.fits',field=(0,1,2))
    #RES_TQU_UNFIL = hp.read_map(cmb_noise_comb_filename, field=(0,1))


    #Qlen_part_FIL,Ulen_part_FIL = TQU_map_FIL[1] * mask, TQU_map_FIL[2] * mask
    Qlen_part_UNFIL,Ulen_part_UNFIL = TQU_map_UNFIL[1] * mask, TQU_map_UNFIL[2] * mask

    #Qlen_noisy_part_FIL, Ulen_noisy_part_FIL = CMB_TQU_FIL[1] * mask, CMB_TQU_FIL[2] * mask
    Qlen_noisy_part_UNFIL, Ulen_noisy_part_UNFIL = CMB_TQU_UNFIL[0] * mask, CMB_TQU_UNFIL[1] * mask

    #Qnoise_part_FIL, Unoise_part_FIL = RES_TQU_FIL[1] * mask, RES_TQU_FIL[2] * mask                                                   #noise QU1
    #Qnoise_part_UNFIL, Unoise_part_UNFIL = RES_TQU_UNFIL[1] * mask, RES_TQU_UNFIL[2] * mask                                            #noise QU2

    Qnoise_part_UNFIL, Unoise_part_UNFIL = Qlen_noisy_part_UNFIL-Qlen_part_UNFIL, Ulen_noisy_part_UNFIL-Ulen_part_UNFIL



    phi_noisy_part = pmap_rec * mask                                          #noisy phi (reconstructed phi, has been WIener filtered)

    Pmap_part = Pmap_full * mask                                              #signal phi

    phi_noise_part = phi_noisy_part - Pmap_part

    """
    phi_noisy_part = (phi_noise+Pmap_full) * mask                                          #noisy phi (reconstructed phi, has been WIener filtered)

    Pmap_part = Pmap_full * mask                                              #signal phi

    phi_noise_part = phi_noise
    """

    ##########################################################################
    ##########################    Wiener filter   ############################
    def do_map_wf(nside, lmax, maps, wfs):
        """
        do map Wiener filter.
        """
        qlm = hp.almxfl(hp.map2alm(maps[0]), wfs)
        ulm = hp.almxfl(hp.map2alm(maps[1]), wfs)

        Q_map = hp.alm2map(qlm,nside,lmax)
        U_map = hp.alm2map(ulm,nside,lmax)

        return Q_map,U_map

    ratio1 = Wiener_filter_QU(nside,lmax,[Qlen_noisy_part_UNFIL, Qlen_noisy_part_UNFIL, Ulen_noisy_part_UNFIL],[Qnoise_part_UNFIL, Qnoise_part_UNFIL, Unoise_part_UNFIL],)

    Qlen_noisy_part_FIL, Ulen_noisy_part_FIL = do_map_wf(nside, lmax,[Qlen_noisy_part_UNFIL, Ulen_noisy_part_UNFIL], ratio1)

    #Qnoise_part_FIL, Unoise_part_FIL = do_map_wf(nside, lmax, [Qnoise_part_UNFIL, Unoise_part_UNFIL], ratio1)

    Qlen_part_FIL,Ulen_part_FIL = do_map_wf(nside, lmax, [Qlen_part_UNFIL, Ulen_part_UNFIL], ratio1)

    Qnoise_part_FIL, Unoise_part_FIL = Qlen_noisy_part_FIL - Qlen_part_FIL, Ulen_noisy_part_FIL - Ulen_part_FIL


    ###########################################################################
    ###############     get deconvolved 'input' noisy maps     

    Bmap_noisy_origin = E2Bcorrection_new(nside,50,[Qlen_noisy_part_UNFIL,Qlen_noisy_part_UNFIL,Ulen_noisy_part_UNFIL], mask, mask, lmax=3000, flag='clean', cal_cl=False, n_iter=4,is_Dell=True)

    Bmap_noise_origin = E2Bcorrection_new(nside,50,[Qnoise_part_UNFIL,Qnoise_part_UNFIL,Unoise_part_UNFIL], mask, mask, lmax=3000, flag='clean', cal_cl=False, n_iter=4, is_Dell=True)


    ###########################################################################
    ########################        Wiener_filter     #########################
    def Wiener_filter_phi_new(nside,lmax,pure_map,noisy_map,):
        plm_pure = curvedsky.utils.hp_map2alm(nside, lmax, lmax, pure_map)
        plm_noisy_full = curvedsky.utils.hp_map2alm(nside, lmax, lmax, noisy_map)
        clpp_pure_full = curvedsky.utils.alm2cl(lmax, plm_pure, alm2=None)
        clpp_noisy_full = curvedsky.utils.alm2cl(lmax, plm_noisy_full, alm2=None)
        ratio = (clpp_pure_full) * cli(clpp_noisy_full)
        return ratio

    #ratio1 = Wiener_filter_QU(nside,lmax,[Qlen_noisy_part, Qlen_noisy_part, Ulen_noisy_part],[Qnoise_part,Qnoise_part,Unoise_part],)
    #ratio2 = Wiener_filter_phi(nside,dlmax,Pmap_part,phi_noisy_part)
    #ratio2 = wiener

    Qlm_noisy = hp.map2alm(Qlen_noisy_part_FIL)
    Ulm_noisy = hp.map2alm(Ulen_noisy_part_FIL)

    Qlm_noise = hp.map2alm(Qnoise_part_FIL)
    Ulm_noise = hp.map2alm(Unoise_part_FIL)

    Qlm_pure = hp.map2alm(Qlen_part_FIL)
    Ulm_pure = hp.map2alm(Ulen_part_FIL)


    Plm_noisy = hp.map2alm(phi_noisy_part, lmax=dlmax)
    Plm_noise = hp.map2alm(phi_noise_part, lmax=dlmax)
    Plm_pure = hp.map2alm(Pmap_part, lmax=dlmax)


    combined_alm_len = QU2comb(nside,lmax, Qlm_noisy, Ulm_noisy)      #Wiener-filter QU noisy
    combined_pure = QU2comb(nside,lmax, Qlm_pure, Ulm_pure)           #Wiener-filter QU signal
    combined_alm_only_noise = QU2comb(nside,lmax, Qlm_noise, Ulm_noise)    #Wiener-filter real QU noise

    Plm_noisy_pix = curvedsky.utils.lm_healpy2healpix(Plm_noisy, dlmax, lmpy=0)  #Wiener-filter phi noisy
    Plm_noise_pix = curvedsky.utils.lm_healpy2healpix(Plm_noise, dlmax, lmpy=0)  #Wiener-filter phi noise
    Plm_pure_pix = curvedsky.utils.lm_healpy2healpix(Plm_pure, dlmax, lmpy=0)    #Wiener-filter phi signal


    ###########################################################################
    ###############      calculating beta and then delensing      #############

    beta_noisy = curvedsky.delens.shiftvec(nside, dlmax, Plm_noisy_pix, nremap=4)
    #beta_noise = curvedsky.delens.shiftvec(nside, lmax, Plm_noise_pix, nremap=4)
    beta_pure = curvedsky.delens.shiftvec(nside, dlmax, Plm_pure_pix, nremap=4)
    
    combined_alm_de = curvedsky.delens.remap_tp(nside, lmax, beta_noisy, combined_alm_len)            #delensed noisy_noisy
    #combined_alm_noise_de1 = curvedsky.delens.remap_tp(nside, lmax, beta_noisy, combined_alm_only_noise)   #delensed noisy_noise
    #combined_alm_noise_de2 = curvedsky.delens.remap_tp(nside, lmax, beta_noise, combined_pure)        #delensed noise_pure
    combined_alm_pure_de = curvedsky.delens.remap_tp(nside, lmax, beta_pure, combined_pure)           #delensed pure_pure

    #combined_alm_bias_fg = curvedsky.delens.remap_tp(nside, lmax, beta_noisy, combined_alm_fg)        #delensed noise_noise


    ###########################################################################
    ##################       noise & signal post process     ##################

    _,Bmap_full = qu2eb_map(nside,lmax,TQU_map_UNFIL[1],TQU_map_UNFIL[2])
    Blen = Bmap_full

    ###########################################################################
    ##################       noise & signal post process     ##################

    
    B_noisy_de = curvedsky.utils.hp_alm2map(nside, lmax, lmax, combined_alm_de[2])         # yes
    B_pure_de =  curvedsky.utils.hp_alm2map(nside, lmax, lmax, combined_alm_pure_de[2])    # yes
    B_noise_de = B_noisy_de - B_pure_de                                                    # yes

    #B_noise_fg_fil = curvedsky.utils.hp_alm2map(nside, lmax, lmax, combined_alm_only_noise[2])
    #B_noise_differ = B_noise_fg_fil - B_noise_de          # NOTICE: THIS IS B_{temp}^{N} = - B_{del}^{N}

    B_pure_temp =  curvedsky.utils.hp_alm2map(nside, lmax, lmax, (combined_pure - combined_alm_pure_de)[2])
    B_noisy_temp = curvedsky.utils.hp_alm2map(nside, lmax, lmax, (combined_alm_len - combined_alm_de)[2])

    B_noise_differ = B_noisy_temp - B_pure_temp
    
    #B_noise_de1 = curvedsky.utils.hp_alm2map(nside, lmax, lmax, combined_alm_noise_de1[2])
    #B_noise_de2 = curvedsky.utils.hp_alm2map(nside, lmax, lmax, (combined_alm_noise_de2 - combined_pure)[2])
    #B_noise_de = B_noise_de1 + B_noise_de2


    ###########################################################################
    ##################       calculate pseudo cl     ##########################
    # Notice: the 'get_pseudo_cl' function will multiply the mask again, so do not multiply the smoothed mask before inputting.


    ######################## For template auto-power spectrum ################
    l_bin,dlbb_temp_noisy = get_pseudo_cl(nside,mask_cut_sm,'00',40,lmax=2000 ,apo=True,f0=[B_noisy_temp,B_noisy_temp],f2=None,is_Dell=True)
    l_bin,dlbb_temp_noise = get_pseudo_cl(nside,mask_cut_sm,'00',40,lmax=2000 ,apo=True,f0=[B_noise_differ,B_noise_differ],f2=None,is_Dell=True)
    l_bin,dlbb_pure = get_pseudo_cl(nside,mask_cut_sm,'00',40,lmax=2000 ,apo=True,f0=[B_pure_temp,B_pure_temp],f2=None,is_Dell=True)

    l_bin,dlbb_temp_cross_bias = get_pseudo_cl(nside,mask_cut_sm,'00',40,lmax=2000 ,apo=True,f0=[B_noise_differ,B_pure_temp],f2=None,is_Dell=True)   #NEW CROSS BIAS DUE TO PHI NOISE
    #################cross term is negligible, so we do not calculate it.################


    ######################## For template cross-power spectrum ################
    l_bin,dlbb_cross_pure = get_pseudo_cl(nside,mask_cut_sm,'00',40,lmax=2000 ,apo=True,f0=[B_pure_temp, Bmap_noisy_origin],f2=None,is_Dell=True)  # pure temp * obs map

    l_bin,dlbb_cross = get_pseudo_cl(nside,mask_cut_sm,'00',40,lmax=2000 ,apo=True,f0=[B_noisy_temp, Bmap_noisy_origin],f2=None,is_Dell=True)  # noisy temp * obs map

    l_bin,dlbb_cross_noise2 = get_pseudo_cl(nside,mask_cut_sm,'00',40,lmax=2000 ,apo=True,f0=[Blen, B_noise_differ],f2=None,is_Dell=True)   # signal * noise temp

    l_bin,dlbb_cross_noise1 = get_pseudo_cl(nside,mask_cut_sm,'00',40,lmax=2000 ,apo=True,f0=[Bmap_noise_origin, B_noisy_temp],f2=None,is_Dell=True)  # noise * noisy temp

    l_bin,dlbb_cross_pure_new = get_pseudo_cl(nside,mask_cut_sm,'00',40,lmax=2000 ,apo=True,f0=[B_pure_temp, Blen],f2=None,is_Dell=True)  # pure temp * obs map

    ######################## For delensed spectrum ################
    #l_bin,dlbb_noisy_de = get_pseudo_cl(nside,mask_cut_sm,'00',20,lmax=2000 ,apo=True,f0=[Bmap_noisy_de,Bmap_noisy_de],f2=None,is_Dell=True)  #noisy delensed map

    #l_bin,dlbb_res = get_pseudo_cl(nside,mask_cut_sm,'00',20,lmax=2000 ,apo=True,f0=[Bmap_pure_de,Bmap_pure_de],f2=None,is_Dell=True)  #pure delensed map

    #l_bin,dlbb_res_noise_differ = get_pseudo_cl(nside,mask_cut_sm,'00',20,lmax=2000 ,apo=True,f0=[Bmap_pure_de,Bmap_noise_differ],f2=None,is_Dell=True)  #noise differ

    #l_bin,dlbb_noise_differ = get_pseudo_cl(nside,mask_cut_sm,'00',20,lmax=2000 ,apo=True,f0=[Bmap_noise_differ,Bmap_noise_differ],f2=None,is_Dell=True)  #noise differ

    #l_bin,dlbb_noise = get_pseudo_cl(nside,mask_cut_sm,'00',20,lmax=2000 ,apo=True,f0=[Bmap_noise_origin,Bmap_noise_origin],f2=None,is_Dell=True)



    Bmap_origin = E2Bcorrection_new(nside,50,[TQU_map_UNFIL[1], TQU_map_UNFIL[1], TQU_map_UNFIL[2]], mask, mask, lmax=3000, flag='clean', cal_cl=False, n_iter=4,is_Dell=True)
    l_bin,dlbb_len_bin = get_pseudo_cl(nside,mask_cut_sm,'00',40,lmax=2000 ,apo=True,f0=[Bmap_origin,Bmap_origin],f2=None,is_Dell=True)
    transfer_temp = dlbb_pure.T[:,0]*cli(dlbb_len_bin[0,:])
    transfer_temp_cross = dlbb_cross_pure[0,:]*cli(dlbb_len_bin[0,:])
    transfer_temp_cross_new = dlbb_cross_pure_new[0,:]*cli(dlbb_len_bin[0,:])

    np.savetxt(cl_auto_filename, [l_bin[:49], dlbb_temp_noisy[0,:49], dlbb_temp_noise[0,:49], dlbb_pure[0,:49], dlbb_temp_cross_bias[0,:49], dlbb_len_bin[0,:49], transfer_temp[:49]], fmt='%f')
    np.savetxt(cl_cross_filename, [l_bin[:49], dlbb_cross_pure[0,:49], dlbb_cross[0,:49], dlbb_cross_noise2[0,:49], dlbb_cross_noise1[0,:49], transfer_temp_cross[:49], transfer_temp_cross_new[:49]], fmt='%f')
    #np.savetxt(cl_de_filename, [l_bin[:49], dlbb_noisy_de[0,:49], dlbb_res[0,:49], dlbb_res_noise_differ[0,:49], dlbb_noise_differ[0,:49], dlbb_noise[0,:49],], fmt='%f')



    ###########################################################################
    ##################       noise & signal post process     ##################
    """
    Bmap_noise =  (Bmap_noise1 + Bmap_noise2)
    Bmap_noise_differ = (Bmap_noise_origin - Bmap_noise)
    Bmap_noisy_de = (Bmap_noisy_origin - Bmap_noisy)

    _,Bmap_full = qu2eb_map(nside,lmax,TQU_map_UNFIL[1],TQU_map_UNFIL[2])
    Bmap_pure_de = (Bmap_full - Bmap_pure)
    Elen,Blen = qu2eb_map(nside,lmax,TQU_map_UNFIL[1],TQU_map_UNFIL[2])


    l_bin, dlbb_noisy_de = get_pseudo_cl(nside,mask_cut_sm,'00',20,lmax=2000 ,apo=True,f0=[Bmap_noisy_de, Bmap_noisy_de],f2=None,is_Dell=True)

    l_bin, dlbb_noise = get_pseudo_cl(nside,mask_cut_sm,'00',20,lmax=2000 ,apo=True,f0=[Bmap_noise_origin, Bmap_noise_origin],f2=None,is_Dell=True)

    l_bin, dlbb_deln = get_pseudo_cl(nside,mask_cut_sm,'00',20,lmax=2000 ,apo=True,f0=[-Bmap_noise, -Bmap_noise],f2=None,is_Dell=True)

    l_bin, dlbb_dels = get_pseudo_cl(nside,mask_cut_sm,'00',20,lmax=2000 ,apo=True,f0=[Bmap_pure_de, Bmap_pure_de],f2=None,is_Dell=True)

    l_bin, dlbb_dels_cross_noise = get_pseudo_cl(nside,mask_cut_sm,'00',20,lmax=2000 ,apo=True,f0=[Bmap_noise_origin, Bmap_pure_de],f2=None,is_Dell=True)  

    l_bin, dlbb_dels_cross_deln = get_pseudo_cl(nside,mask_cut_sm,'00',20,lmax=2000 ,apo=True,f0=[-Bmap_noise, Bmap_pure_de],f2=None,is_Dell=True)

    l_bin, dlbb_noise_cross_deln = get_pseudo_cl(nside,mask_cut_sm,'00',20,lmax=2000 ,apo=True,f0=[-Bmap_noise, Bmap_noise_origin],f2=None,is_Dell=True)

    np.savetxt(cl_terms_filename, [l_bin[:49], dlbb_noisy_de[0,:49], dlbb_noise[0,:49], dlbb_deln[0,:49], dlbb_dels[0,:49], dlbb_dels_cross_noise[0,:49], dlbb_dels_cross_deln[0,:49], dlbb_noise_cross_deln[0,:49],], fmt='%f')
    """
    return None

if __name__ == "__main__":

    nside = 2048

    # 调用 delensing_ones 函数
    delensing_ones(nside,)