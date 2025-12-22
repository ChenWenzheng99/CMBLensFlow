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


from lensingb_mine import *
from utils_mine import *
from lenspyx.utils import camb_clfile
from lenspyx.utils_hp import synalm, almxfl, alm2cl

camb_path = os.path.realpath(os.path.join(os.getcwd(),'..'))    
sys.path.insert(0,camb_path)

import camb
from camb import model, initialpower ,correlations

from scipy.special import factorial

import pymaster as nm


###########################################################################
######################### Some useful functions ###########################
###########################################################################

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

# Obtain the rank and size of the MPI job
args = sys.argv[1:]

# Print the current job's rank
file_indices = comm_rank + int(args[0])

# Print the current job's file index list
print(f"Job {comm_rank} will process files: {file_indices}")


# Input simulation map (signal-only)
TQU_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_ILC/mmDL_maps_2048/{file_indices:05d}/lensedcmb_{file_indices:05d}.fits"
P_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_ILC/mmDL_maps_2048/{file_indices:05d}/kappa_{file_indices:05d}.fits"

# Observed CMB map (noise+signal)
cmb_noisy_comb_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MAIN/revise/rec/prepare/comb_delens_maps/final_obs/obs_final_QU_map_{file_indices:04d}.fits"

# CMB noise (residual) map (noise-only)
cmb_noise_comb_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MAIN/revise/rec/prepare/comb_delens_maps/final_res/res_final_QU_map_{file_indices:04d}.fits"

# Lensing proxy map (from internal reconstruction or external tracer)
phi_noisy_comb_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MAIN/revise/rec/QE_HO_temperature_COMB/qlm_QE_MV_red/qlm_QE_MV_{file_indices:04d}.fits"




# output cl filenames
cl_auto_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MAIN/revise/delens/output_cl_red/temp_QE_ASL/auto/all_in_one_{file_indices}.dat"
cl_cross_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MAIN/revise/delens/output_cl_red/temp_QE_ASL/cross/all_in_one_{file_indices}.dat"
cl_de_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MAIN/revise/delens/output_cl_red/temp_QE_ASL/de/all_in_one_{file_indices}.dat"
cl_obs_filename = f"/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MAIN/revise/delens/output_cl_red/temp_QE_ASL/obs/all_in_one_{file_indices}.dat"


# Load global mask
mask = hp.read_map(f'/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MAIN/sim/mask/ALI_LAT_add_SO_LAT_cross_PLANCK_2048_sm_thin_0.2.fits')
mask_sm = hp.read_map(f'/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MAIN/sim/mask/ALI_LAT_add_SO_LAT_cross_PLANCK_2048_sm_thin_0.2.fits')   #smoothed mask is necessary

mask_cut = hp.read_map(f'/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MAIN/sim/mask/ALI_LAT_add_SO_LAT_cross_PLANCK_2048_sm_thin_0.5.fits')
mask_cut_sm = hp.read_map(f'/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MAIN/sim/mask/ALI_LAT_add_SO_LAT_cross_PLANCK_2048_sm_thin_0.5.fits')   #smoothed mask is necessary


print(f"Job {comm_rank} will write map to {cl_auto_filename}:")


##########################################################################
class delensing_Bremap:
    """
    CMB B-mode delensing with inverse-lensing remapping method.

    Parameters
    ----------
    nside : int
        Healpix nside parameter.
    lmax : int
        Maximum multipole for CMB fields.
    elmin, elmax : int
        Minimum and maximum multipoles for E-mode fields used for constructing the B-mode template.
    dlmin, dlmax : int
        Minimum and maximum multipoles for lensing potential fields used for constructing the B-mode template.
    fil_phi : bool, optional
        Whether to apply filtering to the lensing potential map (default is True). NOTICE: If you are using external tracers (joint tracer) as lensing potential, set fil_phi=False, since which has already been filtered to optimal.
    bin_size : int, optional
        Bin size for power spectrum estimation (default is 40).

    Input CMB maps should be in Healpix FITS format.
    Input lensing potential maps can be in either Healpix FITS format or alm format or .npy format.
    

    NOTICE: 
    * This class is designed for simulation pipelines, where both the input (true), noise (residual) and observed maps are available.

    * This allows for precise separation of signal and noise components in the analysis, to quantify the delensing performance accurately and give forecast to future experiments.

    * FOR REAL DATA, please modify the code accordingly, where the input (true) maps are not available, and noise (residual) cannot be precisely obtained.

    
    """
    def __init__(self,nside,lmax,dlmin,dlmax,fil_phi=True,bin_size=40):
        self.nside = nside
        self.lmax = lmax
        self.dlmin = dlmin
        self.dlmax = dlmax
        self.fil_phi = fil_phi
        self.bin_size = bin_size

        q2k = lambda l: l*(l + 1) / 2 # potential -> convergence
        q2d = lambda l: (l*(l + 1)) ** 0.5 # potential -> deflection

    def load_map(self,TQU_filename,P_filename,cmb_noisy_comb_filename,cmb_noise_comb_filename,phi_noisy_comb_filename):
        print('loading maps...')
        
        # Signal CMB
        self.TQU_map_UNFIL = hp.read_map(TQU_filename,field=(0,1,2))                # TQU map

        # Observed CMB (signal + noise)
        self.CMB_TQU_UNFIL = hp.read_map(cmb_noisy_comb_filename, field=(0,1))      # QU  map

        # Noise CMB (residual noise only)
        #self.RES_TQU_UNFIL = hp.read_map(cmb_noise_comb_filename, field=(0,1))     # QU  map

        # Lensing potential maps
        try:
            self.plm_input = hp.map2alm(hp.read_map(P_filename), self.dlmax)        # phi alm
        except:
            self.plm_input = hp.read_alm(P_filename)

        # Reconstructed lensing potential (noisy)
        fname = phi_noisy_comb_filename
        if fname.endswith(".fits"):                                                 # phi alm
            try:
                self.plm_rec = hp.read_alm(fname)
            except Exception:
                phi_map = hp.read_map(fname)
                self.plm_rec = hp.map2alm(phi_map, lmax=self.dlmax)

        elif fname.endswith(".npy"):
            self.plm_rec = np.load(fname)

        else:
            raise ValueError(f"Unknown phi file format: {fname}")

        print('maps loaded.')

    def process_phi(self):
        """
        Process lensing potential alm maps with Wiener filtering.
        """
        print('processing phi alm...')
        if self.fil_phi:
            print('Wiener filtering for phi alm...')
            # Wiener filter for lensing potential alm
            clpp_rec = (hp.alm2cl(self.plm_rec)/np.mean(mask))
            wiener = hp.alm2cl(self.plm_input)[:len(clpp_rec)]/clpp_rec
            wiener[np.where(wiener == np.inf)] = 0
            wiener[0] = 0
        else:
            print('No more Wiener filtering for phi alm...')
            wiener = np.ones(self.dlmax + 1)
        cut = np.where((np.arange(self.dlmax + 1) > self.dlmin) * (np.arange(self.dlmax + 1) < self.dlmax), 1, 0)      # band limit, remove very low and very high multipoles

        pmap_input = hp.alm2map(hp.almxfl(self.plm_input, cut * wiener), self.nside,)
        pmap_rec = hp.alm2map(hp.almxfl(self.plm_rec, cut * wiener), self.nside)

        self.phi_noisy_part = pmap_rec * mask                                          #noisy phi (reconstructed phi, has been WIener filtered)

        self.Pmap_part = pmap_input * mask                                              #signal phi

        self.phi_noise_part = self.phi_noisy_part - self.Pmap_part

        print('phi processed.')

    def process_CMB(self):
        """
        Do Wiener filter to CMB Q/U maps.
        """
        print('processing CMB maps...')

        self.Qlen_part_UNFIL, self.Ulen_part_UNFIL = self.TQU_map_UNFIL[1] * mask, self.TQU_map_UNFIL[2] * mask                                                     # signal CMB Q/U maps                          

        self.Qlen_noisy_part_UNFIL, self.Ulen_noisy_part_UNFIL = self.CMB_TQU_UNFIL[0] * mask, self.CMB_TQU_UNFIL[1] * mask                                         # observed CMB Q/U maps (signal + noise)

        self.Qnoise_part_UNFIL, self.Unoise_part_UNFIL = self.Qlen_noisy_part_UNFIL - self.Qlen_part_UNFIL, self.Ulen_noisy_part_UNFIL - self.Ulen_part_UNFIL       # noise Q/U maps


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

        ratio1 = Wiener_filter_QU(self.nside,self.lmax,[self.Qlen_noisy_part_UNFIL, self.Qlen_noisy_part_UNFIL, self.Ulen_noisy_part_UNFIL],[self.Qnoise_part_UNFIL, self.Qnoise_part_UNFIL, self.Unoise_part_UNFIL],)

        self.Qlen_noisy_part_FIL, self.Ulen_noisy_part_FIL = do_map_wf(self.nside, self.lmax,[self.Qlen_noisy_part_UNFIL, self.Ulen_noisy_part_UNFIL], ratio1)           # observed CMB Q/U maps after Wiener filter (signal + noise)

        self.Qlen_part_FIL,self.Ulen_part_FIL = do_map_wf(self.nside, self.lmax, [self.Qlen_part_UNFIL, self.Ulen_part_UNFIL], ratio1)                                   # signal CMB Q/U maps after Wiener filter

        self.Qnoise_part_FIL, self.Unoise_part_FIL = self.Qlen_noisy_part_FIL - self.Qlen_part_FIL, self.Ulen_noisy_part_FIL - self.Ulen_part_FIL                        # noise Q/U maps after Wiener filter

        print('CMB maps processed.')


    def prep_map2alm(self):
        """
        Convert Q/U maps to E/B alms and lensing potential maps to alm, for B-mode template construction.
        """
        Qlm_noisy = hp.map2alm(self.Qlen_noisy_part_FIL)
        Ulm_noisy = hp.map2alm(self.Ulen_noisy_part_FIL)

        Qlm_noise = hp.map2alm(self.Qnoise_part_FIL)
        Ulm_noise = hp.map2alm(self.Unoise_part_FIL)

        Qlm_pure = hp.map2alm(self.Qlen_part_FIL)
        Ulm_pure = hp.map2alm(self.Ulen_part_FIL)

        Plm_noisy = hp.map2alm(self.phi_noisy_part)
        Plm_noise = hp.map2alm(self.phi_noise_part)
        Plm_pure = hp.map2alm(self.Pmap_part)


        # convert Q/U alm to E/B alm
        self.combined_alm_len = QU2comb(self.nside,self.lmax, Qlm_noisy, Ulm_noisy)      #Wiener-filter QU noisy
        self.combined_pure = QU2comb(self.nside,self.lmax, Qlm_pure, Ulm_pure)           #Wiener-filter QU signal
        self.combined_alm_only_noise = QU2comb(self.nside,self.lmax, Qlm_noise, Ulm_noise)    #Wiener-filter real QU noise

        self.Plm_noisy_pix = curvedsky.utils.lm_healpy2healpix(Plm_noisy, self.dlmax, lmpy=0)  #Wiener-filter phi noisy
        self.Plm_noise_pix = curvedsky.utils.lm_healpy2healpix(Plm_noise, self.dlmax, lmpy=0)  #Wiener-filter phi noise
        self.Plm_pure_pix = curvedsky.utils.lm_healpy2healpix(Plm_pure, self.dlmax, lmpy=0)    #Wiener-filter phi signal


    def construct_Btemp(self):
        """
        Construct B-mode lensing template from E-mode and lensing potential maps.

        See Eq.(A.3) of https://arxiv.org/abs/2507.19897 for reference.
        """

        print('constructing delensed B mode...')


        ###########################################################################
        ###############      calculating beta and then delensing      #############

        # beta: anti-lensing deflection field
        beta_noisy = curvedsky.delens.shiftvec(self.nside, self.dlmax, self.Plm_noisy_pix, nremap=4)
        #beta_noise = curvedsky.delens.shiftvec(self.nside, self.lmax, self.Plm_noise_pix, nremap=4)
        beta_pure = curvedsky.delens.shiftvec(self.nside, self.dlmax, self.Plm_pure_pix, nremap=4)
        
        self.combined_alm_de = curvedsky.delens.remap_tp(self.nside, self.lmax, beta_noisy, self.combined_alm_len)            #delensed noisy_noisy
        #combined_alm_noise_de1 = curvedsky.delens.remap_tp(self.nside, self.lmax, beta_noisy, self.combined_alm_only_noise)  #delensed noisy_noise
        #combined_alm_noise_de2 = curvedsky.delens.remap_tp(self.nside, self.lmax, beta_noise, self.combined_pure)            #delensed noise_pure
        self.combined_alm_pure_de = curvedsky.delens.remap_tp(self.nside, self.lmax, beta_pure, self.combined_pure)           #delensed pure_pure

        #combined_alm_bias_fg = curvedsky.delens.remap_tp(nside, lmax, beta_noisy, combined_alm_fg)        #delensed noise_noise


        ############################################################################
        #################     Do EB correction for input maps    ###################   

        self.Bmap_noisy_origin = E2Bcorrection_new(self.nside,50,[self.Qlen_noisy_part_UNFIL,self.Qlen_noisy_part_UNFIL,self.Ulen_noisy_part_UNFIL], mask, mask, lmax=3000, flag='clean', cal_cl=False, n_iter=4,is_Dell=True)

        self.Bmap_noise_origin = E2Bcorrection_new(self.nside,50,[self.Qnoise_part_UNFIL,self.Qnoise_part_UNFIL,self.Unoise_part_UNFIL], mask, mask, lmax=3000, flag='clean', cal_cl=False, n_iter=4, is_Dell=True)

        _,self.Blen = qu2eb_map(self.nside,self.lmax,self.TQU_map_UNFIL[1],self.TQU_map_UNFIL[2])

        ###########################################################################
        ##################       noise & signal post process     ##################

        self.B_noisy_de = curvedsky.utils.hp_alm2map(self.nside, self.lmax, self.lmax, self.combined_alm_de[2])         # yes
        self.B_pure_de =  curvedsky.utils.hp_alm2map(self.nside, self.lmax, self.lmax, self.combined_alm_pure_de[2])    # yes
        self.B_noise_de = self.B_noisy_de - self.B_pure_de                                                    # yes

        #B_noise_fg_fil = curvedsky.utils.hp_alm2map(self.nside, self.lmax, self.lmax, self.combined_alm_only_noise[2])
        #B_noise_differ = B_noise_fg_fil - self.B_noise_de          # NOTICE: THIS IS B_{temp}^{N} = - B_{del}^{N}

        self.B_pure_temp =  curvedsky.utils.hp_alm2map(self.nside, self.lmax, self.lmax, (self.combined_pure - self.combined_alm_pure_de)[2])
        self.B_noisy_temp = curvedsky.utils.hp_alm2map(self.nside, self.lmax, self.lmax, (self.combined_alm_len - self.combined_alm_de)[2])
        self.B_noise_differ = self.B_noisy_temp - self.B_pure_temp
        print('B mode template constructed.')


    def compute_cl(self,):
        """
        Compute Cls of observational, template and delensed B maps.
        These will then be used for MCMC fitting.

        See Appendix.A of https://arxiv.org/abs/2507.19897 for reference.
        """

        ###########################################################################
        ##################       calculate pseudo cl     ##########################


        ######################## For template auto-power spectrum ################      (Eq.A4)
        l_bin,dlbb_temp_noisy = get_pseudo_cl(self.nside,mask_cut_sm,'00',self.bin_size,lmax=2000 ,apo=True,f0=[self.B_noisy_temp,self.B_noisy_temp],f2=None,is_Dell=True)          # C^temp
        l_bin,dlbb_temp_noise = get_pseudo_cl(self.nside,mask_cut_sm,'00',self.bin_size,lmax=2000 ,apo=True,f0=[self.B_noise_differ,self.B_noise_differ],f2=None,is_Dell=True)      # ⟨B^tempN B^tempN ⟩
        l_bin,dlbb_pure = get_pseudo_cl(self.nside,mask_cut_sm,'00',self.bin_size,lmax=2000 ,apo=True,f0=[self.B_pure_temp,self.B_pure_temp],f2=None,is_Dell=True)                  # ⟨B^tempS B^tempS ⟩
        l_bin,dlbb_temp_cross_bias = get_pseudo_cl(self.nside,mask_cut_sm,'00',self.bin_size,lmax=2000 ,apo=True,f0=[self.B_noise_differ,self.B_pure_temp],f2=None,is_Dell=True)    # ⟨B^tempS B^tempN ⟩


        ######################## For template cross-power spectrum ################
        l_bin,dlbb_cross_pure = get_pseudo_cl(self.nside,mask_cut_sm,'00',self.bin_size,lmax=2000 ,apo=True,f0=[self.B_pure_temp, self.Bmap_noisy_origin],f2=None,is_Dell=True)      # pure temp * obs map
        l_bin,dlbb_cross = get_pseudo_cl(self.nside,mask_cut_sm,'00',self.bin_size,lmax=2000 ,apo=True,f0=[self.B_noisy_temp, self.Bmap_noisy_origin],f2=None,is_Dell=True)          # C^cross
        l_bin,dlbb_cross_noise2 = get_pseudo_cl(self.nside,mask_cut_sm,'00',self.bin_size,lmax=2000 ,apo=True,f0=[self.Blen, self.B_noise_differ],f2=None,is_Dell=True)              # ⟨B^lens B^tempN ⟩
        l_bin,dlbb_cross_noise1 = get_pseudo_cl(self.nside,mask_cut_sm,'00',self.bin_size,lmax=2000 ,apo=True,f0=[self.Bmap_noise_origin, self.B_noisy_temp],f2=None,is_Dell=True)   # ⟨Bres,NILC Btemp⟩
        l_bin,dlbb_cross_pure_new = get_pseudo_cl(self.nside,mask_cut_sm,'00',self.bin_size,lmax=2000 ,apo=True,f0=[self.B_pure_temp, self.Blen],f2=None,is_Dell=True)               # ⟨B^lens B^tempS ⟩


        ######################## For delensed spectrum ################
        #l_bin,dlbb_noisy_de = get_pseudo_cl(self.nside,mask_cut_sm,'00',self.bin_size,lmax=2000 ,apo=True,f0=[self.Bmap_noisy_de,self.Bmap_noisy_de],f2=None,is_Dell=True)  #noisy delensed map
        #l_bin,dlbb_res = get_pseudo_cl(self.nside,mask_cut_sm,'00',self.bin_size,lmax=2000 ,apo=True,f0=[self.Bmap_pure_de,self.Bmap_pure_de],f2=None,is_Dell=True)  #pure delensed map
        #l_bin,dlbb_res_noise_differ = get_pseudo_cl(self.nside,mask_cut_sm,'00',self.bin_size,lmax=2000 ,apo=True,f0=[self.Bmap_pure_de,self.Bmap_noise_differ],f2=None,is_Dell=True)  #noise differ
        #l_bin,dlbb_noise_differ = get_pseudo_cl(self.nside,mask_cut_sm,'00',self.bin_size,lmax=2000 ,apo=True,f0=[self.Bmap_noise_differ,self.Bmap_noise_differ],f2=None,is_Dell=True)  #noise differ
        #l_bin,dlbb_noise = get_pseudo_cl(self.nside,mask_cut_sm,'00',self.bin_size,lmax=2000 ,apo=True,f0=[self.Bmap_noise_origin,self.Bmap_noise_origin],f2=None,is_Dell=True)


        ################# For observation spectrum #################
        l_bin,dlbb_obs_noisy = get_pseudo_cl(self.nside,mask,'00',self.bin_size,lmax=2000 ,apo=True,f0=[self.Bmap_noisy_origin,self.Bmap_noisy_origin],f2=None,is_Dell=True)  #noisy obs map
        l_bin,dlbb_obs_noise = get_pseudo_cl(self.nside,mask,'00',self.bin_size,lmax=2000 ,apo=True,f0=[self.Bmap_noise_origin,self.Bmap_noise_origin],f2=None,is_Dell=True)  #noise obs map


        ######################## For transfer function ################
        Bmap_origin = E2Bcorrection_new(self.nside,50,[self.TQU_map_UNFIL[1], self.TQU_map_UNFIL[1], self.TQU_map_UNFIL[2]], self.mask, self.mask, lmax=3000, flag='clean', cal_cl=False, n_iter=4,is_Dell=True)
        l_bin,dlbb_len_bin = get_pseudo_cl(self.nside,self.mask_cut_sm,'00',self.bin_size,lmax=2000 ,apo=True,f0=[Bmap_origin,Bmap_origin],f2=None,is_Dell=True)
        transfer_temp = dlbb_pure.T[:,0]*cli(dlbb_len_bin[0,:])
        transfer_temp_cross = dlbb_cross_pure[0,:]*cli(dlbb_len_bin[0,:])
        transfer_temp_cross_new = dlbb_cross_pure_new[0,:]*cli(dlbb_len_bin[0,:])


        ####################### # Save Cls to file #########################
        np.savetxt(cl_auto_filename, [l_bin[:49], dlbb_temp_noisy[0,:49], dlbb_temp_noise[0,:49], dlbb_pure[0,:49], dlbb_temp_cross_bias[0,:49], dlbb_len_bin[0,:49], transfer_temp[:49]], fmt='%f')
        np.savetxt(cl_cross_filename, [l_bin[:49], dlbb_cross_pure[0,:49], dlbb_cross[0,:49], dlbb_cross_noise2[0,:49], dlbb_cross_noise1[0,:49], transfer_temp_cross[:49], transfer_temp_cross_new[:49]], fmt='%f')
        #np.savetxt(cl_de_filename, [l_bin[:49], dlbb_noisy_de[0,:49], dlbb_res[0,:49], dlbb_res_noise_differ[0,:49], dlbb_noise_differ[0,:49], dlbb_noise[0,:49],], fmt='%f')
        np.savetxt(cl_obs_filename, [l_bin[:49], dlbb_obs_noisy[0,:49], dlbb_obs_noise[0,:49], dlbb_len_bin[0,:49], ], fmt='%f')



if __name__ == "__main__":
    # Initialize delensing class
    delens = delensing_Bremap(nside=2048, lmax=6143, dlmin=10, dlmax=5000, fil_phi=True, bin_size=40)

    # Load maps
    delens.load_map(TQU_filename,P_filename,cmb_noisy_comb_filename,cmb_noise_comb_filename,phi_noisy_comb_filename)

    # Process lensing potential maps
    delens.process_phi()

    # Process CMB maps
    delens.process_CMB()

    # Prepare alms for B-mode template construction
    delens.prep_map2alm()

    # Construct B-mode lensing template
    delens.construct_Btemp()

    # Compute Cls
    delens.compute_cl()