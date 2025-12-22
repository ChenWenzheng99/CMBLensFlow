import healpy as hp
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy import units as u
from astropy.cosmology import Planck15
from scipy.interpolate import griddata

from spectral import *


def cal_ilc_bias(lmax, fsky, delta, Nfreq, Ndeproj):
    ell = np.arange(lmax+1)
    Nmodes = 2 * (ell+1) * (delta+1) * fsky

    return np.abs(Ndeproj + 1 - Nfreq) / Nmodes   # see Eq.(35) of arxiv:2307.01043v3


class standard_ILC():
    
    def __init__(self, deltal, spectra, alms, lmax, freqs, fsky):
        """
        Peform standard ILC cleaning.

        Inputs:
        deltal: binning size in ell space, set to O(10) ~ O(100). Important for covariance matrix estimation to suppress ILC bias.
        spectra: instance of the Spectra class. See spectral.py for details.
        alms: list of input alms at different frequencies. Length should be the same as [len(freqs), len(alms)].
        lmax: maximum ell to consider.
        freqs: list of frequencies in GHz. 
        fsky: sky fraction used for Cl estimation. Not important here.

        Outputs:
        alm_clean: cleaned alm after ILC.
        weight: ILC weights as a function of frequency and ell. Shape is [len(freqs), lmax+1].
        """

        self.deltal = deltal
        self.spectra = spectra
        self.alms = alms
        self.ellmax = lmax
        self.ells = np.arange(lmax+1)
        self.freqs = freqs
        self.len = len(self.freqs)
        self.fsky = fsky
        self.Clij = np.zeros((self.len, self.len, self.ellmax+1))
        self.epsilon = 1e-10
    
    def get_Cl(self):
        #Clij = np.zeros((self.len, self.len, self.lmax+1))
        for i in range(self.len):
            for j in range(self.len):
                pre_Cl = hp.alm2cl(self.alms[i], self.alms[j], lmax = self.ellmax) / self.fsky
                for l in range(self.ellmax+1):
                    self.Clij[i][j][l] = pre_Cl[l]

        return self.Clij
    
    def get_Rlij(self):
        self.get_Cl()
        #Construct Rlij covariance matrix
        prefactor = (2*self.ells+1)/(4*np.pi)
        Rlij_no_binning = np.einsum('l,ijl->ijl', prefactor, self.Clij)
        self.Rlij = np.zeros((len(self.freqs), len(self.freqs), self.ellmax+1)) 
        #deltal_array = np.abs(self.freqs-1-self.Ndeproj)/2/self.Rtol/self.fsky/(self.ells+1)
        for i in range(len(self.freqs)):
            for j in range(len(self.freqs)):
                self.Rlij[i][j] = (np.convolve(Rlij_no_binning[i][j], np.ones(2*self.deltal+1)))[self.deltal:self.ellmax+1+self.deltal]
        return self.Rlij
    
    def get_Rlij_inv(self): 
        #Get inverse of R_{\ell}^{ij '}
        #self.Rlij_inv = np.array([np.linalg.pinv(self.Rlij[:,:,l]) for l in range(self.ellmax+1)]) 
        self.Rlij_inv = np.array([np.linalg.inv(self.Rlij[:,:,l]+ self.epsilon * np.eye(self.Rlij[:,:,l].shape[0])) for l in range(self.ellmax+1)]) 
        return self.Rlij_inv #index as Rlij_inv[l][i][j]

    def get_ab(self):
        #get spectral response vectors
        self.a = np.ones(len(self.freqs)) #index as a[i]
        self.b = self.spectra.tsz_spectral_response(self.freqs) #index as b[i]
        return self.a, self.b
    
    
    def weights(self):
        #get weights
        numerator = np.einsum('lij,j->il', self.Rlij_inv, self.a)
        denominator = np.einsum('lkm,k,m->l', self.Rlij_inv, self.a, self.a)
        self.w = numerator/denominator #index as w[i][l]
        print(print('constraint:', np.dot(self.w[:, 100],self.a)))
        return self.w
    
    def do_CILC(self):
        self.get_Rlij()
        self.get_Rlij_inv()
        self.get_ab()
        self.weights()
        weight = self.w / np.sum(self.w, axis=0)   # I think we need to confirm the normalization here again: \Sum_i w_i = 1  !!!!

        alm_clean = np.zeros(len(self.alms[0]), dtype = complex)
        #for l in range(self.ellmax+1):
        #    for m in range(l+1):
        #        j = hp.sphtfunc.Alm.getidx(self.ellmax, l, m)

        for i in range(self.len):
            alm_clean += hp.almxfl(self.alms[i], weight[i],)

        return alm_clean, weight
        


class ILC_deproj_tSZ():
    
    def __init__(self, deltal, spectra, alms, lmax, freqs, fsky):
        """
        Same as standard_ILC, but deprojecting tSZ signal. 

        Be sure to check the tSZ spectral response in spectral.py.
        """

        self.deltal = deltal
        self.spectra = spectra
        self.alms = alms
        self.ellmax = lmax
        self.ells = np.arange(lmax+1)
        self.freqs = freqs
        self.len = len(self.freqs)
        self.fsky = fsky
        self.Clij = np.zeros((self.len, self.len, self.ellmax+1))
        self.epsilon = 1e-10


    def get_Cl(self):
        #Clij = np.zeros((self.len, self.len, self.lmax+1))
        for i in range(self.len):
            for j in range(self.len):
                pre_Cl = hp.alm2cl(self.alms[i], self.alms[j], lmax = self.ellmax) / self.fsky
                for l in range(self.ellmax+1):
                    self.Clij[i][j][l] = pre_Cl[l]

        return self.Clij
    
    def get_Rlij(self):
        self.get_Cl()
        #Construct Rlij covariance matrix
        prefactor = (2*self.ells+1)/(4*np.pi)
        Rlij_no_binning = np.einsum('l,ijl->ijl', prefactor, self.Clij)
        self.Rlij = np.zeros((len(self.freqs), len(self.freqs), self.ellmax+1)) 
        for i in range(len(self.freqs)):
            for j in range(len(self.freqs)):
                self.Rlij[i][j] = (np.convolve(Rlij_no_binning[i][j], np.ones(2*self.deltal+1)))[self.deltal:self.ellmax+1+self.deltal]
        return self.Rlij
    
    def get_Rlij_inv(self): 
        #Get inverse of R_{\ell}^{ij '}
        #self.Rlij_inv = np.array([np.linalg.pinv(self.Rlij[:,:,l]) for l in range(self.ellmax+1)]) 
        self.Rlij_inv = np.array([np.linalg.inv(self.Rlij[:,:,l]+ self.epsilon * np.eye(self.Rlij[:,:,l].shape[0])) for l in range(self.ellmax+1)]) 
        return self.Rlij_inv #index as Rlij_inv[l][i][j]

    def get_ab(self):
        #get spectral response vectors
        self.a = np.ones(len(self.freqs)) #index as a[i]
        self.b = self.spectra.tsz_spectral_response(self.freqs) #index as b[i]
        return self.a, self.b
    
    def get_ABD(self):
        self.A = np.einsum('lij,i,j->l', self.Rlij_inv, self.a, self.a)
        self.B = np.einsum('lij,i,j->l', self.Rlij_inv, self.b, self.b)
        self.D = np.einsum('lij,i,j->l', self.Rlij_inv, self.a, self.b)
        return self.A, self.B, self.D
    
    def weights(self):
        #get weights
        numerator = np.einsum('lij,l,i->jl', self.Rlij_inv, self.B, self.a) \
                    - np.einsum('lij,l,i->jl', self.Rlij_inv, self.D, self.b)
        denominator = np.einsum('l,l->l', self.A, self.B) - np.einsum('l,l->l', self.D, self.D)
        self.w = numerator/denominator #index as w[i][l]
        print(print('constraint:', np.dot(self.w[:, 100],self.a), np.dot(self.w[:, 100],self.b)))
        return self.w
    
    def do_CILC(self):
        self.get_Rlij()
        self.get_Rlij_inv()
        self.get_ab()
        self.get_ABD()
        self.weights()
        weight = self.w / np.sum(self.w, axis=0)   # I think we need to confirm the normalization here again: \Sum_i w_i = 1  !!!!

        alm_clean = np.zeros(len(self.alms[0]), dtype = complex)
        #for l in range(self.ellmax+1):
        #    for m in range(l+1):
        #        j = hp.sphtfunc.Alm.getidx(self.ellmax, l, m)

        for i in range(self.len):
            alm_clean += hp.almxfl(self.alms[i], weight[i],)

        return alm_clean, weight
    


class ILC_deproj_CIB():
    """
    Same as standard_ILC, but deprojecting CIB signal.

    Be sure to check (maybe need modify) the CIB spectral response in spectral.py.
    """
    
    def __init__(self, deltal, spectra, alms, lmax, freqs, fsky):

        self.deltal = deltal
        self.spectra = spectra
        self.alms = alms
        self.ellmax = lmax
        self.ells = np.arange(lmax+1)
        self.freqs = freqs
        self.len = len(self.freqs)
        self.fsky = fsky
        self.Clij = np.zeros((self.len, self.len, self.ellmax+1))
        self.epsilon = 1e-10


    def get_Cl(self):
        #Clij = np.zeros((self.len, self.len, self.lmax+1))
        for i in range(self.len):
            for j in range(self.len):
                pre_Cl = hp.alm2cl(self.alms[i], self.alms[j], lmax = self.ellmax) / self.fsky
                for l in range(self.ellmax+1):
                    self.Clij[i][j][l] = pre_Cl[l]

        return self.Clij
    
    def get_Rlij(self):
        self.get_Cl()
        #Construct Rlij covariance matrix
        prefactor = (2*self.ells+1)/(4*np.pi)
        Rlij_no_binning = np.einsum('l,ijl->ijl', prefactor, self.Clij)
        self.Rlij = np.zeros((len(self.freqs), len(self.freqs), self.ellmax+1)) 
        for i in range(len(self.freqs)):
            for j in range(len(self.freqs)):
                self.Rlij[i][j] = (np.convolve(Rlij_no_binning[i][j], np.ones(2*self.deltal+1)))[self.deltal:self.ellmax+1+self.deltal]
        return self.Rlij
    
    def get_Rlij_inv(self): 
        #Get inverse of R_{\ell}^{ij '}
        #self.Rlij_inv = np.array([np.linalg.pinv(self.Rlij[:,:,l]) for l in range(self.ellmax+1)]) 
        self.Rlij_inv = np.array([np.linalg.inv(self.Rlij[:,:,l]+ self.epsilon * np.eye(self.Rlij[:,:,l].shape[0])) for l in range(self.ellmax+1)]) 
        return self.Rlij_inv #index as Rlij_inv[l][i][j]

    def get_ab(self):
        #get spectral response vectors
        self.a = np.ones(len(self.freqs)) #index as a[i]
        self.b = self.spectra.cib_spectral_response(self.freqs) #index as b[i]
        return self.a, self.b
    
    def get_ABD(self):
        self.A = np.einsum('lij,i,j->l', self.Rlij_inv, self.a, self.a)
        self.B = np.einsum('lij,i,j->l', self.Rlij_inv, self.b, self.b)
        self.D = np.einsum('lij,i,j->l', self.Rlij_inv, self.a, self.b)
        return self.A, self.B, self.D
    
    def weights(self):
        #get weights
        numerator = np.einsum('lij,l,i->jl', self.Rlij_inv, self.B, self.a) \
                    - np.einsum('lij,l,i->jl', self.Rlij_inv, self.D, self.b)
        denominator = np.einsum('l,l->l', self.A, self.B) - np.einsum('l,l->l', self.D, self.D)
        self.w = numerator/denominator #index as w[i][l]
        print(print('constraint:', np.dot(self.w[:, 100],self.a), np.dot(self.w[:, 100],self.b)))
        return self.w
    
    def do_CILC(self):
        self.get_Rlij()
        self.get_Rlij_inv()
        self.get_ab()
        self.get_ABD()
        self.weights()
        weight = self.w / np.sum(self.w, axis=0)   # I think we need to confirm the normalization here again: \Sum_i w_i = 1  !!!!

        alm_clean = np.zeros(len(self.alms[0]), dtype = complex)
        #for l in range(self.ellmax+1):
        #    for m in range(l+1):
        #        j = hp.sphtfunc.Alm.getidx(self.ellmax, l, m)

        for i in range(self.len):
            alm_clean += hp.almxfl(self.alms[i], weight[i],)

        return alm_clean, weight
    


class ILC_deproj_tSZ_and_CIB():
    """
    Same as standard_ILC, but deprojecting both tSZ and CIB signals.

    Be sure to check the spectral responses in spectral.py.
    """
    
    def __init__(self, deltal, spectra, alms, lmax, freqs, fsky):
        self.deltal = deltal
        self.spectra = spectra
        self.alms = alms
        self.ellmax = lmax
        self.ells = np.arange(lmax+1)
        self.freqs = freqs
        self.len = len(self.freqs)
        self.fsky = fsky
        self.Clij = np.zeros((self.len, self.len, self.ellmax+1))
        self.epsilon = 1e-10

    def get_Cl(self):
        #Clij = np.zeros((self.len, self.len, self.lmax+1))
        for i in range(self.len):
            for j in range(self.len):
                pre_Cl = hp.alm2cl(self.alms[i], self.alms[j], lmax = self.ellmax) / self.fsky
                for l in range(self.ellmax+1):
                    self.Clij[i][j][l] = pre_Cl[l]
        return self.Clij
    
    def get_Rlij(self):
        self.get_Cl()
        #Construct Rlij covariance matrix
        prefactor = (2*self.ells+1)/(4*np.pi)
        Rlij_no_binning = np.einsum('l,ijl->ijl', prefactor, self.Clij)
        self.Rlij = np.zeros((len(self.freqs), len(self.freqs), self.ellmax+1)) 
        for i in range(len(self.freqs)):
            for j in range(len(self.freqs)):
                self.Rlij[i][j] = (np.convolve(Rlij_no_binning[i][j], np.ones(2*self.deltal+1)))[self.deltal:self.ellmax+1+self.deltal]
        return self.Rlij
    
    def get_Rlij_inv(self): 
        #Get inverse of R_{\ell}^{ij '}
        #self.Rlij_inv = np.array([np.linalg.inv(self.Rlij[:,:,l]) for l in range(self.ellmax+1)]) 
        #self.Rlij_inv = np.array([np.linalg.pinv(self.Rlij[:,:,l]) for l in range(self.ellmax+1)]) 
        self.Rlij_inv = np.array([np.linalg.inv(self.Rlij[:,:,l]+ self.epsilon * np.eye(self.Rlij[:,:,l].shape[0])) for l in range(self.ellmax+1)]) 
        return self.Rlij_inv #index as Rlij_inv[l][i][j]
    
    def get_abc(self):
        #get spectral response vectors
        self.a = np.ones(len(self.freqs)) #index as a[i]
        self.b = self.spectra.tsz_spectral_response(self.freqs) #index as b[i]
        self.c = self.spectra.cib_spectral_response(self.freqs) #index as c[i]
        return self.a, self.b, self.c
    
    def get_ABCDEFQ(self):
        self.A = np.einsum('lij,i,j->l', self.Rlij_inv, self.a, self.a)
        self.B = np.einsum('lij,i,j->l', self.Rlij_inv, self.b, self.b)
        self.C = np.einsum('lij,i,j->l', self.Rlij_inv, self.c, self.c)
        self.D = np.einsum('lij,i,j->l', self.Rlij_inv, self.a, self.b)
        self.E = np.einsum('lij,i,j->l', self.Rlij_inv, self.a, self.c)
        self.F = np.einsum('lij,i,j->l', self.Rlij_inv, self.b, self.c)
        self.Q = np.einsum('l,l,l->l', self.A, self.B, self.C) + 2*np.einsum('l,l,l->l', self.D, self.E, self.F) \
            - np.einsum('l,l,l->l', self.A, self.F, self.F) - np.einsum('l,l,l->l', self.B, self.E, self.E) \
            - np.einsum('l,l,l->l', self.C, self.D, self.D)
        self.Q[0] = 1
        return self.A, self.B, self.C, self.D, self.E, self.F, self.Q
    
    def weights(self):
        #Define weights
        self.w = np.einsum('lij,l,l,l,i->jl', self.Rlij_inv, 1/self.Q, self.B, self.C, self.a) \
            - np.einsum('lij,l,l,l,i->jl', self.Rlij_inv, 1/self.Q, self.F, self.F, self.a) \
            + np.einsum('lij,l,l,l,i->jl', self.Rlij_inv, 1/self.Q, self.E, self.F, self.b) \
            - np.einsum('lij,l,l,l,i->jl', self.Rlij_inv, 1/self.Q, self.C, self.D, self.b) \
            + np.einsum('lij,l,l,l,i->jl', self.Rlij_inv, 1/self.Q, self.D, self.F, self.c) \
            - np.einsum('lij,l,l,l,i->jl', self.Rlij_inv, 1/self.Q, self.B, self.E, self.c)
        print(print('constraint:', np.dot(self.w[:, 100],self.a), np.dot(self.w[:, 100],self.b), np.dot(self.w[:, 100],self.c)))
        return self.w
    
    def do_CILC(self):
        self.get_Rlij()
        self.get_Rlij_inv()
        self.get_abc()
        self.get_ABCDEFQ()
        self.weights()
        weight = self.w / np.sum(self.w, axis=0)   # I think we need to confirm the normalization here again: \Sum_i w_i = 1  !!!!
        print(weight.shape)

        alm_clean = np.zeros(len(self.alms[0]), dtype = complex)
        #for l in range(self.ellmax+1):
        #    for m in range(l+1):
        #        j = hp.sphtfunc.Alm.getidx(self.ellmax, l, m)

        for i in range(self.len):
            alm_clean += hp.almxfl(self.alms[i], weight[i],)

        return alm_clean, weight
    


def do_CILC(freqs, alms, weight, ellmax):
    """
    A supplementary function to perform the given weights on the alms.
    """
    alm_clean = np.zeros(len(alms[0]), dtype = complex)
    for i in range(len(alms)):
        alm_clean += hp.almxfl(alms[i], weight[i],)
    return alm_clean