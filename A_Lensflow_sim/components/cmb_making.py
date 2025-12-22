#load modules
print('loading modules...')
import sys
sys.path.append('/root/download/cmblensplus2/utils')

import numpy as np
import healpy as hp
import pymaster as nmt
import matplotlib.pyplot as plt

##########################################################################

import os,sys
import pylab as pl
import numpy as np
import lenspyx
import healpy as hp

from utils_mine import *

from lenspyx.utils import camb_clfile
from lenspyx.utils_hp import synalm, almxfl, alm2cl

camb_path = os.path.realpath(os.path.join(os.getcwd(),'..'))    
sys.path.insert(0,camb_path)

import camb
from camb import model, initialpower ,correlations


from scipy.special import factorial


def load_file(fn, lmax=None, ifield=0):
    if fn.endswith('.npy'):
       return np.load(fn)[:None]
    elif fn.endswith('.fits'):
        return hp.read_map(fn, field=ifield)
    elif fn.endswith('.dat'):
        return camb_clfile(fn)

def making_maps_new(nside, fid_source='FFP10', fid_type='cl', lmax=None, dlmax=None, seed=None, epsilon=None, if_lensing=True, unlensed_dir=None, phi_dir=None, CAMB_params=None, lenspyx_geom='healpix'):
    """
    Gnerate CMB maps with or without lensing, as well as the lensing map.
    
    Parameters
    ----------
    nside : int
        Resolution of the map
    
    fid_source : str
        Source of the fiducial data. 'FFP10', 'CAMB'(set to Planck by default) or 'custom' (given the unlensed CMB data dir). Default is 'FFP10'.
    
    fid_type : str
        Type of the fiducial data. 'cl' or 'map' or 'alm'. Default is 'cl'. (Only needed when 'custom' is on.)

    lmax : int
        Maximum multipole of the output maps. Default is 3*nside-1.
    
    dlmax : int
        Maximum multipole of the lensing potential, had better to be larger than lmax (+ 512 ~ 1024). Default is (lmax + 512).
    
    seed : int
        Seed for the random number generator (useless when 'custom' is on). Default is A RANDOM INTEGER.

    epsilon : float
        Target accuracy of the output maps (execution time has a fairly weak dependence on this). Default is 1e-6.
    
    if_lensing : bool
        If True, lensing will be applied to the CMB maps. Default is True.

    unlensed_dir : array
        The directory of the provided unlensed CMB map. Default is None. (only needed when both 'custom' and 'if_lensing' are on.)

    phi_dir : array
        The directory of the provided lensing potential map. Default is None. (only needed when both 'custom' and 'if_lensing' are on.)

    CAMB_params : dict
        The parameters for CAMB. Default is None. (only needed when 'fid_source' is 'CAMB'.)

    lenspyx_geom : str
        The geometry on which to produce the lensed map with lenspyx. Default is 'healpix'. Can can be 'healpix', 'thingauss', etc.
        SEE : '/root/anaconda3/envs/lens/lib/python3.10/site-packages/lenspyx/remapping/utils_geom.py' for more details.
    -------

    Notice:
    ****************  If you choose 'custom' as the fid_source, you need to provide the directory of the unlensed CMB map and the lensing potential map.
    ****************  If you choose 'CAMB' as the fid_source, you need to provide the CAMB_params. Otherwise, the default CAMB_params will be used.
    ****************  The 'FFP10' is the fiducial data from Planck 2018, and the 'CAMB' is the fiducial data from CAMB.
    ****************  We have tested that the consistency between potential cl from 'FFP10' and potential cl from  'CAMB' (difference freaction ~ 1e-3 -> 1e-2)
    ****************  But there does exist some difference between the CMB cls from 'FFP10' and 'CAMB' (difference freaction ~ 1e-2 -> 1e-1)

    """
    print('Setting parameters...')
    if lmax is None:
        lmax = 3*nside-1  
    lmax_len = lmax         # desired lmax of the lensed field.

    if dlmax is None:
        dlmax = lmax + 512      

    if nside <= 1024:
        lmax_unl, mmax_unl = lmax_len + dlmax, lmax_len + dlmax   # lmax of the unlensed fields is lmax + dlmax.  (some buffer is required for accurate lensing at lmax)
    else:
        lmax_unl, mmax_unl = 7000, 7000   # This is limited by the lmax of FFP10_wdipole_lenspotentialCls.dat, you may change it if you have higher lmax power spectrum.

    if seed == None:
        seed = np.random.randint(0,99999999)

    if epsilon is None:
        epsilon = 1e-6  # target accuracy of the output maps (execution time has a fairly weak dependence on this)


    if fid_source == 'FFP10':
        cl = camb_clfile('./cls/FFP10_wdipole_lenspotentialCls.dat', lmax_unl)   #l,TT,EE,BB,TE,PP,TP,EP
    elif fid_source == 'CAMB':
        lmax_unl, mmax_unl = lmax_len + dlmax, lmax_len + dlmax   # This will be not limited by FFP10 lmax.
        clpp, unlens_cls = get_cls_from_CAMB(lmax_unl, CAMB_params)
        cl = {'tt':unlens_cls[:,0], 'ee':unlens_cls[:,1], 'bb':unlens_cls[:,2], 'te':unlens_cls[:,3], 'pp':clpp}
    elif fid_source == 'custom':
        assert unlensed_dir is not None and phi_dir is not None, 'Please provide the directory of the unlensed CMB map and the lensing potential map.'


    #unlensed CMB SPHERICAL HARMONICS COEFFICIENTS
    np.random.seed(seed)
    if fid_source == 'FFP10' or fid_source == 'CAMB':
        tlm_unl, elm_unl, blm_unl = hp.synalm([cl['tt'], cl['te'], cl['tt']*0, cl['ee'], cl['tt']*0, cl['bb']], lmax=lmax_unl, mmax=mmax_unl)
        # tlm_unl = hp.synalm(cl['tt'], lmax=lmax_unl, mmax=mmax_unl)
        # elm_unl = hp.synalm(cl['ee'], lmax=lmax_unl, mmax=mmax_unl)
        # blm_unl = hp.synalm(cl['bb'], lmax=lmax_unl, mmax=mmax_unl)
    elif fid_source == 'custom':
        unlensed_map = load_file(unlensed_dir, lmax=lmax_unl, ifield=(0,1,2))
        tlm_unl = hp.map2alm(unlensed_map[0], lmax=lmax_unl)
        elm_unl, blm_unl = hp.map2alm_spin(unlensed_map[1:], 2, lmax=lmax_unl)


    if not if_lensing:
        print('No lensing...')
        Tunl = hp.alm2map(tlm_unl, nside, lmax=lmax_unl, )
        Qunl, Uunl = hp.alm2map_spin([elm_unl, blm_unl], nside, 2, lmax=lmax_unl)
        print('Unlensed CMB Map generating done !!!')
        return Tunl, Qunl, Uunl, np.zeros_like(Tunl)
    else:
        if fid_source == 'custom':
            pmap = load_file(phi_dir, lmax=dlmax, ifield=0)  
            plm = hp.map2alm(pmap, lmax=dlmax, )
        else:
            np.random.seed(seed+999999)
            plm = hp.synalm(cl['pp'],lmax=dlmax,new=True,)
            pmap = hp.alm2map(plm, nside, lmax=dlmax, )

        print('Lensing...')
        # We then transform the lensing potential into spin-1 deflection field, (see below:  deflect the temperature map.)
        dlm = almxfl(plm, np.sqrt(np.arange(dlmax + 1, dtype=float) * np.arange(1, dlmax + 2)), None, False)  

        # Geometry on which to produce the lensed map
        if lenspyx_geom == 'healpix':
            geom_info = ('healpix', {'nside':nside})
        elif lenspyx_geom == 'thingauss':
            geom_info = ('thingauss',{'lmax': lmax, 'smax': 1})

        ######geom_info = ('healpix', {'nside':nside}) # here we will use an Healpix grid with nside 2048
        Tlen, Qlen, Ulen = lenspyx.alm2lenmap([tlm_unl, elm_unl, blm_unl], dlm, geometry=geom_info, verbose=1, epsilon=epsilon)

        # Check if the simulation is corrupted, sometimes the simulation will be corrupted, be sure to check it.
        if np.any(pmap > 1) or np.any(Tlen > 1000) or np.any(Qlen > 100) or np.any(Ulen > 100) :
            print(f'Warning : simulation {seed} has corrupted, need to simulate in the next round !')

        print('Lensed CMB Map generating done !!!')
        return Tlen, Qlen, Ulen, pmap




def get_cls_from_CAMB(lmax, CAMB_params=None):
    #Temporary used for generating the input Cls
    #Run Camb，give the theoretical results
    print('running CAMB...')
    Tcmb = 2.7255e6

    # Default cosmological parameters (Planck 2018)
    default_params = {           # Parameters from planck 'FFP10_wdipole_params.ini'
        "H0": 67.01904,          # Hubble constant (H_0, km/s/Mpc)
        "ombh2": 0.02216571,     # Baryon density fraction (\Omega_b h^2)
        "omch2": 0.1202944,      # Cold dark matter density fraction (\Omega_c h^2)
        "mnu": 0.0006451439,     # Neutralino mass (m_\nu h^2)
        "omk": 0,                # Curvature parameter (\Omega_k)
        "tau": 0.06018107,       # Reionization optical depth (\tau)
        "YHe": 0.2453006,        # Helium mass fraction (Y_p)
        "w": -1.0,               # Dark energy equation of state
        "wa": 0,                 # Dark energy equation of state slope
        "nnu": 3.046,            # Effective number of neutrinos
        "nt": 0,                 # Primordial tensor power law tilt
        "nrun": 0,               # Primordial running of the scalar spectral index
        "pivot_scalar": 0.05,    # Scalar spectrum pivot point
        "pivot_tensor": 0.05,    # Tensor spectrum pivot point
        "As": 2.119631e-09,      # Scalar perturbation amplitude
        "ns": 0.9636852,         # Scalar power law tilt
        "r": 0,                  # Tensor-to-scalar ratio
        "lens_potential_accuracy": 8,  # Lensing potential calculation accuracy
    }

    # If CAMB_params is None, set it to an empty dictionary
    if CAMB_params is None:
        CAMB_params = {}

    # Update the default parameters with the user-provided parameters
    for key, value in default_params.items():
        if key not in CAMB_params:
            CAMB_params[key] = value


    # Set up a new set of parameters for CAMB
    # Object to store the parameters
    pars = camb.CAMBparams()
    #This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency

    pars = camb.set_params(H0=CAMB_params['H0'], ombh2=CAMB_params['ombh2'], omch2=CAMB_params['omch2'], mnu=CAMB_params['mnu'], omk=CAMB_params['omk'], tau=CAMB_params['tau'], 
                           YHe=CAMB_params['YHe'], w=CAMB_params['w'], wa=CAMB_params['wa'], nnu=CAMB_params['nnu'], nt=CAMB_params['nt'], nrun=CAMB_params['nrun'], 
                           pivot_scalar=CAMB_params['pivot_scalar'], pivot_tensor=CAMB_params['pivot_tensor'])
    
    # Turn on tensor modes
    pars.WantTensors = True

    # Set initial power spectrum parameters
    pars.InitPower.set_params(As=CAMB_params['As'], ns=CAMB_params['ns'], r=CAMB_params['r'])

    # Set accuracy and lmax parameters for lensing potential calulation
    pars.set_for_lmax(lmax, lens_potential_accuracy=8)

    #calculate results for these parameters
    results = camb.get_results(pars)

    ##############################################################################
    #生成时无量纲，对CMB auto spectrum 要乘Tcmb**2, 对lensing potential auto spectrum 保持无量纲, 对CMB-potential cross spectrum 乘Tcmb
    cl_ppte=results.get_lens_potential_cls(lmax=None, CMB_unit='muK', raw_cl=True)   #numpy array CL[0:lmax+1,0:3], where 0..2 indexes PP, PT, PE.(T,E均unlensed)
    clpp=cl_ppte[:,0]

    unlensed_total = results.get_unlensed_total_cls(raw_cl=True) * Tcmb**2 # unlensed CMB power spectra, including tensors

    #dl_ppte=results.get_lens_potential_cls(lmax=None, CMB_unit='muK', raw_cl=False)   #numpy array CL[0:lmax+1,0:3], where 0..2 indexes PP, PT, PE.(T,E均unlensed)
    #dlpp=dl_ppte[:,0]
    #lensed_cl = results.get_lensed_cls_with_spectrum(dlpp, lmax=None, CMB_unit=None, raw_cl=True) * Tcmb**2

    return clpp, unlensed_total







"""
def get_cls_from_CAMB(lmax, r=0):

    print('running CAMB...')
    Tcmb = 2.7255e6

    # Set cosmological parameters
    params = camb.set_params(H0=67.01904, ombh2=0.02216571, omch2=0.1202944, omk=0, tau=0.06018107, YHe=0.2453006, w=-1.0, wa=0, nt=-0/8.0, nrun=0,\
                        pivot_scalar=0.05, pivot_tensor=0.05, )
    params.omegan = 0.0006451439 / (67.01904**2 / 100**2)  # Convert omnuh2 to omegan

    # Set dark energy parameters
    params.set_dark_energy(w=-1.0)

    # Set initial power spectrum parameters
    params.InitPower.set_params(As=2.119631e-9, ns=0.9636852, nrun=0.0)

    # Set accuracy and lmax parameters
    params.set_for_lmax(lmax=lmax, lens_potential_accuracy=10, max_eta_k=100000)

    # Set additional accuracy parameters
    params.Accuracy.AccuracyBoost = 2
    params.Accuracy.lAccuracyBoost = 2
    params.Accuracy.HighAccuracyDefault = True


    # Set other parameters
    params.WantCls = True
    params.Want_CMB = True
    params.WantTransfer = False
    params.WantTensors = False
    params.WantVectors = False
    params.WantScalars = True
    params.NonLinear = model.NonLinear_both
    params.DoLensing = True
    params.Alens = 1.0
    params.DoLateRadTruncation = True
    params.WantDerivedParameters = True

    # Reionization settings
    params.Reion.Reionization = True
    params.Reion.use_optical_depth = True
    params.Reion.optical_depth = 0.06018107
    params.Reion.delta_redshift = 0.5
    params.Reion.fraction = -1
    params.Reion.Helium_redshift = 3.5
    params.Reion.Helium_delta_redshift = 0.5
    params.Reion.Helium_redshiftstart = 5.0




    # Neutrino settings
    # Ensure other parameters are correctly set
    params.TCMB = 2.7255
    params.YHe = 0.2453006
    params.num_massless_neutrinos = 2.03066666667
    params.num_massive_neutrinos = 1
    params.nu_mass_fractions = [1.0]
    params.nu_mass_degeneracies = [1.01533333333]
    params.share_delta_neff = False

    #calculate results for these parameters
    results = camb.get_results(params)

    ##############################################################################
    cl_ppte=results.get_lens_potential_cls(lmax=None, CMB_unit='muK', raw_cl=True)   #numpy array CL[0:lmax+1,0:3], where 0..2 indexes PP, PT, PE.(T,E均unlensed)
    clpp=cl_ppte[:,0]

    unlensed_total = results.get_unlensed_total_cls(raw_cl=True) * Tcmb**2 # unlensed CMB power spectra, including tensors

    #dl_ppte=results.get_lens_potential_cls(lmax=None, CMB_unit='muK', raw_cl=False)   #numpy array CL[0:lmax+1,0:3], where 0..2 indexes PP, PT, PE.(T,E均unlensed)
    #dlpp=dl_ppte[:,0]
    #lensed_cl = results.get_lensed_cls_with_spectrum(dlpp, lmax=None, CMB_unit=None, raw_cl=True) * Tcmb**2

    return clpp, unlensed_total

"""