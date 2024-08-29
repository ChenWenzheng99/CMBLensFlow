import os
import sys
import healpy as hp
import numpy as np

sys.path.insert(0, './')
from one import ALILENS
from library_parameter import *
from utils import bl

class simsLensing:    #用于读图，sims分成CMB,NOISE等部分，可以自己添加：residual, foreground, point source等，得到最终模拟的图，对应实际观测图。
    def __init__(self):
        self.cmbs = os.path.join( ALILENS_cmb, 'map_TQU_2048_%04d.fits')  #cmbs: path of simulated lensed CMB map
        #self.cmbs = os.path.join( ALILENS_cmb, f'map_TQU_2048_{idx+100:04d}.fits')  #cmbs: path of simulated lensed CMB map  #cmbs: path of simulated lensed CMB map

    def hashdict(self):
        return {'cmbs': self.cmbs}   #返回一个包含模拟数据哈希值的字典

    
    def replace_fwhm_alm(self,alm,fwhm_old,fwhm_new,pix_old,pix_new,lmax):
        """
        replace old beam of alm with new beam in arcmin
        """
        bl_old = bl(fwhm_old, nside=nside, lmax=lmax, pixwin=pix_old)     #sim.py生成的有pixwin,sim_ali.py生成的没有
        bl_new = bl(fwhm_new, nside=nside, lmax=lmax, pixwin=pix_new)    #还原为没有beam和pixwin
        alm_deconv = hp.almxfl(alm,1/bl_old * bl_new)
        return alm_deconv

    def replace_fwhm_map(self,nside,lmax,map,fwhm_old,fwhm_new,pix_old,pix_new,):
        """
        replace old beam of map with new beam in arcmin
        """
        alm = hp.map2alm(map)
        alm_deconv = self.replace_fwhm_alm(alm,fwhm_old,fwhm_new,pix_old,pix_new,lmax)
        map_deconv = hp.alm2map(alm_deconv,nside)
        return map_deconv

    def get_sim_tmap(self, idx):
        T = hp.read_map(self.cmbs % (idx+300), field=0)
        #ac2rad = np.pi/10800.
        T = self.replace_fwhm_map(nside, lmax, T, 1.4, 2.2, True, True)
        noise = hp.read_map( os.path.join(ALILENS_noise, f'map_noise_nside2048_{idx+300:04d}.fits') , field=0) #simulated noise Tmap
        fg = hp.read_map( os.path.join(ALILENS_fg, f'IQU_FG_{idx+300}.fits') , field=0)
        fg = self.replace_fwhm_map(nside, lmax, fg, 0, 2.2, False, True)
        return T + noise + fg #final sim lensed Tmap

    def get_sim_pmap(self, idx):
        Q, U = hp.read_map(self.cmbs % (idx+300), field=(1,2))
        #ac2rad = np.pi/10800.
        Q = self.replace_fwhm_map(nside, lmax, Q, 1.4, 2.2, True, True)
        U = self.replace_fwhm_map(nside, lmax, U, 1.4, 2.2, True, True)
        noise_Q, noise_U = hp.read_map( os.path.join(ALILENS_noise, f'map_noise_nside2048_{idx+300:04d}.fits') , field=(1,2)) #simulated noise Qmap and Umap
        fg_Q, fg_U = hp.read_map( os.path.join(ALILENS_fg, f'IQU_FG_{idx+300}.fits'), field=(1,2))
        fg_Q = self.replace_fwhm_map(nside, lmax, fg_Q, 0, 2.2, False, True)
        fg_U = self.replace_fwhm_map(nside, lmax, fg_U, 0, 2.2, False, True)
        return Q + noise_Q + fg_Q, U + noise_U + fg_U #final sim lensed Qmap and Umap
