import os,sys

current_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = '../A_Lensflow_sim'  
full_path = os.path.join(current_dir, relative_path)
sys.path.append(full_path)

from components.cmb_making import *
from components.foreground_making import *
from components.noise_making import *


import healpy as hp
import numpy as np

class SkySimulator:
    def __init__(self, config):
        """
        Initializing the simulator

        Args:
            config (dict): A dictionary containing simulation configuration parameters.
        """
        self.config = config

    def simulate(self):
        """生成模拟天图，根据配置组合不同成分。"""
        nside = self.config['nside']
        lmax = self.config['lmax']
        freqs = self.config['freqs']
        beam = self.config['beam']
        mask_dir = self.config['mask_dir']
        if mask_dir is not None:
            mask = hp.read_map(mask_dir)
        else:
            mask = np.ones(hp.nside2npix(nside))


        ######################### CMB #########################  [T,Q,U]
        if self.config.get('cmb', False):
            print('-----------------------------------------------------------------')
            print('****************  Simulating CMB ****************')
            fid_type = self.config['cmb_params']['fid_type']
            fid_source = self.config['cmb_params']['fid_source']
            dlmax = self.config['cmb_params']['dlmax']
            seed = self.config['cmb_params']['seed']
            epsilon = self.config['cmb_params']['epsilon']
            if_lensing = self.config['cmb_params']['if_lensing']
            phi_dir = self.config['cmb_params']['phi_dir']
            unlensed_dir = self.config['cmb_params']['unlensed_dir']
            CAMB_params = self.config['cmb_params']['CAMB_params']
            lenspyx_geom = self.config['cmb_params']['lenspyx_geom']


            cmb_phi_map = making_maps_new(nside, fid_source=fid_source, fid_type=fid_type, lmax=lmax, dlmax=dlmax, seed=seed, epsilon=epsilon, if_lensing=if_lensing, phi_dir=phi_dir, unlensed_dir=unlensed_dir, CAMB_params=CAMB_params,lenspyx_geom='healpix')
            cmb_map = np.array([cmb_phi_map[0], cmb_phi_map[1], cmb_phi_map[2]])
            phi_map = cmb_phi_map[3]
        else:
            cmb_map = np.zeros((3, hp.nside2npix(nside))) 

        ######################### Foreground #########################  [freqs, [T,Q,U]]
        if self.config.get('foreground', False):
            print('-----------------------------------------------------------------')
            print('****************  Simulating Foreground ****************')
            comps = self.config['foreground_params']['comps']
            freqs = self.config['freqs']
            unit = self.config['foreground_params']['unit']
            cordinate = self.config['foreground_params']['cordinate']
            if_gaussian = self.config['foreground_params']['if_gaussian']
            seed = self.config['foreground_params']['seed']

            fg_maps = []
            if not if_gaussian:
                for i in range(len(freqs)):
                    print('Simulating Foreground at', freqs[i], 'GHz')
                    fg_map = pysm_generator(nside, comps, freqs[i], unit=unit, cordinate=cordinate)  
                    fg_maps.append(fg_map)
            else:
                for i in range(len(freqs)):
                    print('Generating Gaussian Foreground at', freqs[i], 'GHz')
                    nus = {freqs[i]: freqs[i]*1e9}
                    fg_map,_ = gaussian_fg(nside, nus, seed[i], same=False, method=1, A_d_b = 14.30, alpha_d_b = -0.65, A_s_b = 2.40, alpha_s_b = -0.80, beta_d = 1.48, beta_s = -3.10, eps_ds = 0,
                            A_d_e = 24.00, alpha_d_e = -0.60, A_s_e = 5.10, alpha_s_e = -0.88, A_d_t = 220.00, alpha_d_t = -0.60, A_s_t = 5.10, alpha_s_t = -0.83,)  
                    fg_maps.append(np.array([fg_map[freqs[i]][0], fg_map[freqs[i]][1], fg_map[freqs[i]][2]]))
        else:
            fg_maps = np.zeros((len(freqs), 3, hp.nside2npix(nside)))
        fg_maps = np.array(fg_maps)
        print('-----------------------------------------------------------------')

        ######################### Noise #########################   [freqs, [T,Q,U]]
        if self.config.get('noise', False):
            print('****************  Simulating Noise ****************')
            nlev = self.config['noise_params']['nlev']
            noise_model = self.config['noise_params']['noise_model']
            seed = self.config['noise_params']['seed']


            N_red_T = self.config['noise_params']['N_red_T']
            if N_red_T is None:
                N_red_T = [None] * len(freqs)
            l_knee_T = self.config['noise_params']['l_knee_T']
            if l_knee_T is None:
                l_knee_T = [None] * len(freqs)
            alpha_knee_T = self.config['noise_params']['alpha_knee_T']
            if alpha_knee_T is None:
                alpha_knee_T = [None] * len(freqs)
            n_red_P = self.config['noise_params']['n_red_P']
            if n_red_P is None:
                n_red_P = [None] * len(freqs)
            l_knee_P = self.config['noise_params']['l_knee_P']
            if l_knee_P is None:
                l_knee_P = [None] * len(freqs)
            alpha_knee_P = self.config['noise_params']['alpha_knee_P']
            if alpha_knee_P is None:
                alpha_knee_P = [None] * len(freqs)

            if_red = self.config['noise_params']['if_red']
            scale_factor = self.config['noise_params']['scale_factor']

            noise_maps = []
            for i in range(len(freqs)):
                print('Simulating Noise at', freqs[i], 'GHz')
                noise_map = call_noise_model(nside, noise_model, nlev[i], seed[i], N_red_T[i], l_knee_T[i], alpha_knee_T[i], n_red_P[i], l_knee_P[i], alpha_knee_P[i], if_red, scale_factor)
                noise_maps.append(noise_map)
        else:
            noise_maps = np.zeros((len(freqs), 3, hp.nside2npix(nside)))
        noise_maps = np.array(noise_maps)
        print('-----------------------------------------------------------------')
        print('****************  Combining Maps ****************')

        def bl(fwhm, lmax=None, nside=None, pixwin=True):    #bl includes beam window function (e^-(l(l+1)* \sigma^2)) and pixel window function
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


        def add_beam(fwhm_ac,nside,maps,lmax=None,pixwin=False):
            """
            Smooth the maps with beam fwhm_ac in arcmin and pixwin. 
            
            Better for full sky input CMB/FG maps (which will be convolved with beam and pixwin).
            """
            if lmax == None:
                lmax =3*nside-1
            transf = bl(fwhm_ac, nside=nside, lmax=lmax, pixwin=pixwin)
            tlm_len = hp.map2alm(maps[0], lmax=lmax)
            elm_len, blm_len = hp.map2alm_spin([maps[1], maps[2]], 2, lmax=lmax)
            Tlen = hp.alm2map(hp.almxfl(tlm_len, transf, inplace=True), nside, verbose=False)
            Qlen, Ulen = hp.alm2map_spin([hp.almxfl(elm_len, transf), hp.almxfl(blm_len, transf)], nside, 2, lmax)
            return Tlen,Qlen,Ulen  

        
        total_maps = []
        for i in range(len(freqs)):
            print('Simulating Observed Sky at', freqs[i], 'GHz')
            total_map = (add_beam(beam[i], nside, cmb_map + fg_maps[i], lmax, pixwin=False ) + noise_maps[i] ) * mask
            total_maps.append(total_map)

        print('****************  Simulation Finished ****************')
        if self.config['cmb_params']['if_lensing'] == True:
            return total_maps, phi_map
        else:
            return total_maps

