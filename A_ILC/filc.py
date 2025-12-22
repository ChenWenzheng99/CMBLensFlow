import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pandas as pd
import gc
import symlens as s
from pathlib import Path


import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pandas as pd
import gc

from pathlib import Path
from pixell import enmap, enplot, utils, coordinates, reproject, wcsutils, powspec, lensing, curvedsky
from scipy.interpolate import interp1d

def cal_ilc_bias(lmax, fsky, delta, Nfreq, Ndeproj):
    ell = np.arange(lmax+1)
    Nmodes = 2 * (ell+1) * (delta+1) * fsky

    return np.abs(Ndeproj + 1 - Nfreq) / Nmodes   # see Eq.(35) of arxiv:2307.01043v3


def invert_covariance(R, epsilon=1e-6):
    """
    R: (n_band, n_band, nx, ny) 协方差矩阵场
    epsilon: 正则化强度
    返回: R_inv, 形状同 (n_band, n_band, nx, ny)
    """
    nb, _, nx, ny = R.shape
    N = nx * ny

    # reshape -> (N, nb, nb)
    R_batch = np.transpose(R.reshape(nb, nb, N), (2,0,1))

    # 加 epsilon I
    R_batch_reg = R_batch + epsilon * np.eye(nb)[None,:,:]

    # 单位矩阵 batch
    I = np.eye(nb)[None,:,:].repeat(N, axis=0)

    # 逐个 solve (等价于 inv(R) @ I)
    R_inv_batch = np.linalg.solve(R_batch_reg, I)   # shape (N, nb, nb)

    # 转回 (nb, nb, nx, ny)
    R_inv = np.transpose(R_inv_batch, (1,2,0)).reshape(nb, nb, nx, ny)

    return R_inv


class standard_ILC():
    
    def __init__(self, deltal, spectra, kmaps, lmax, freqs, fsky):
        """
        Standard ILC implementation in Fourier space.

        Inputs:
        kmaps: list of enmap, shape as (nfreq, ny, nx)
        freqs: list of frequencies, shape as (nfreq,)
        spectra: instance of Spectral_response class
        deltal: bin width for power spectrum smoothing
        lmax: maximum ell
        fsky: sky fraction

        Outputs:
        kmap_out: enmap of the ILC cleaned map
        weight: enmap of the weights applied at each frequency
        """

        self.deltal = deltal
        self.spectra = spectra
        self.kmaps = kmaps
        self.shape = kmaps[0].shape
        self.wcs = kmaps[0].wcs
        self.ellmax = lmax
        self.ells = np.arange(lmax+1)
        self.freqs = freqs
        self.len = len(self.freqs)
        self.fsky = fsky
        self.Clij = np.zeros((self.len, self.len, self.shape[0], self.shape[1]))
        self.epsilon = 1e-10
    
    def get_Cl(self):
        #Clij = np.zeros((self.len, self.len, self.lmax+1))
        for i in range(self.len):
            for j in range(self.len):
                pre_Cl = np.real(self.kmaps[i] * np.conj(self.kmaps[j]))
                self.Clij[i][j] = pre_Cl

        return self.Clij
    
    def isotropic_2d_from_Cl(self, ell, Cl, fill_value="extrapolate"):
        Lx, Ly  = enmap.lmap(self.shape, self.wcs)
        Lmod    = np.hypot(Lx, Ly)

        # 用线性插值把 1D C_ell 映射到每个像素的 |l|
        f = interp1d(ell, Cl, kind="linear", bounds_error=False, fill_value=fill_value)
        S2d = f(Lmod)
        return enmap.enmap(S2d, self.wcs)

    def Smooth_power_2D(self, kpower, delta_ell):
        modlmap = enmap.modlmap(self.shape, self.wcs)
        bin_edges = np.arange(2, self.ellmax, self.deltal)
        binner = s.bin2D(modlmap,bin_edges)
        
        cents, nl1d_real = binner.bin(np.real(kpower))
        S2d = self.isotropic_2d_from_Cl(cents, nl1d_real)

        return S2d
    
    def get_Rlij(self):
        self.get_Cl()
        self.Rlij = np.zeros_like(self.Clij)
        for i in range(len(self.freqs)):
            for j in range(len(self.freqs)):
                self.Rlij[i][j] = self.Smooth_power_2D(self.Clij[i][j], self.deltal)
        return self.Rlij
    
    def get_Rlij_inv(self): 
        try:
            self.Rlij_inv = invert_covariance(self.Rlij, epsilon=0)
        except np.linalg.LinAlgError:
            print("Matrix is singular, considering normalization with epsilon=1e-10.")
            self.Rlij_inv = invert_covariance(self.Rlij, epsilon=1e-10)
        return self.Rlij_inv #index as Rlij_inv[l][i][j]
    
    def get_ab(self):
        #get spectral response vectors
        self.a = np.ones(len(self.freqs)) #index as a[i]
        self.b = self.spectra.tsz_spectral_response(self.freqs) #index as b[i]
        return self.a, self.b
    
    def weights(self):
        #get weights
        numerator = np.einsum('ijpx,j->ipx', self.Rlij_inv, self.a)
        denominator = np.einsum('kmpx,k,m->px', self.Rlij_inv, self.a, self.a)
        self.w = numerator/denominator #index as w[i][l]
        print(print('constraint:', np.dot(self.w[:, 10, 10],self.a)))
        return self.w
    
    def do_CILC(self):
        self.get_Rlij()
        self.get_Rlij_inv()
        self.get_ab()
        self.weights()
        weight = self.w / np.sum(self.w, axis=0)   # I think we need to confirm the normalization here again: \Sum_i w_i = 1  !!!!

        kmap_out = enmap.zeros(self.shape, self.wcs, dtype=np.complex128)
        #for l in range(self.ellmax+1):
        #    for m in range(l+1):
        #        j = hp.sphtfunc.Alm.getidx(self.ellmax, l, m)

        for i in range(self.len):
            kmap_out += weight[i] * self.kmaps[i]

        return kmap_out, weight
    


class ILC_deproj_tSZ():
    
    def __init__(self, deltal, spectra, kmaps, lmax, freqs, fsky):
        """
        Same as standard_ILC, but deprojecting tSZ signal.

        Be sure to check the tSZ spectral response in spectral.py.
        """

        self.deltal = deltal
        self.spectra = spectra
        self.kmaps = kmaps
        self.shape = kmaps[0].shape
        self.wcs = kmaps[0].wcs
        self.ellmax = lmax
        self.ells = np.arange(lmax+1)
        self.freqs = freqs
        self.len = len(self.freqs)
        self.fsky = fsky
        self.Clij = np.zeros((self.len, self.len, self.shape[0], self.shape[1]))
        self.epsilon = 1e-10
    
    def get_Cl(self):
        #Clij = np.zeros((self.len, self.len, self.lmax+1))
        for i in range(self.len):
            for j in range(self.len):
                pre_Cl = np.real(self.kmaps[i] * np.conj(self.kmaps[j]))
                self.Clij[i][j] = pre_Cl

        return self.Clij
    
    def isotropic_2d_from_Cl(self, ell, Cl, fill_value="extrapolate"):
        Lx, Ly  = enmap.lmap(self.shape, self.wcs)
        Lmod    = np.hypot(Lx, Ly)

        # 用线性插值把 1D C_ell 映射到每个像素的 |l|
        f = interp1d(ell, Cl, kind="linear", bounds_error=False, fill_value=fill_value)
        S2d = f(Lmod)
        return enmap.enmap(S2d, self.wcs)

    def Smooth_power_2D(self, kpower, delta_ell):
        modlmap = enmap.modlmap(self.shape, self.wcs)
        bin_edges = np.arange(2, self.ellmax, self.deltal)
        binner = s.bin2D(modlmap,bin_edges)
        
        cents, nl1d_real = binner.bin(np.real(kpower))
        S2d = self.isotropic_2d_from_Cl(cents, nl1d_real)

        return S2d
    
    def get_Rlij(self):
        self.get_Cl()
        self.Rlij = np.zeros_like(self.Clij)
        for i in range(len(self.freqs)):
            for j in range(len(self.freqs)):
                self.Rlij[i][j] = self.Smooth_power_2D(self.Clij[i][j], self.deltal)
        return self.Rlij
    
    def get_Rlij_inv(self): 
        try:
            self.Rlij_inv = invert_covariance(self.Rlij, epsilon=0)
        except np.linalg.LinAlgError:
            print("Matrix is singular, considering normalization with epsilon=1e-10.")
            self.Rlij_inv = invert_covariance(self.Rlij, epsilon=1e-10)
        return self.Rlij_inv #index as Rlij_inv[l][i][j]
    
    def get_ab(self):
        #get spectral response vectors
        self.a = np.ones(len(self.freqs)) #index as a[i]
        self.b = self.spectra.tsz_spectral_response(self.freqs) #index as b[i]
        return self.a, self.b
    
    def get_ABD(self):
        self.A = np.einsum('ijpx,i,j->px', self.Rlij_inv, self.a, self.a)
        self.B = np.einsum('ijpx,i,j->px', self.Rlij_inv, self.b, self.b)
        self.D = np.einsum('ijpx,i,j->px', self.Rlij_inv, self.a, self.b)
        return self.A, self.B, self.D
    
    def weights(self):
        #get weights
        numerator = np.einsum('ijpx,px,i->jpx', self.Rlij_inv, self.B, self.a) \
                    - np.einsum('ijpx,px,i->jpx', self.Rlij_inv, self.D, self.b)
        denominator = np.einsum('px,px->px', self.A, self.B) - np.einsum('px,px->px', self.D, self.D)
        self.w = numerator/denominator #index as w[i][l]
        print(print('constraint:', np.dot(self.w[:, 10, 10],self.a), np.dot(self.w[:, 10, 10],self.b)))
        return self.w
    
    def do_CILC(self):
        self.get_Rlij()
        self.get_Rlij_inv()
        self.get_ab()
        self.get_ABD()
        self.weights()
        weight = self.w / np.sum(self.w, axis=0)   # I think we need to confirm the normalization here again: \Sum_i w_i = 1  !!!!

        kmap_out = enmap.zeros(self.shape, self.wcs, dtype=np.complex128)
        #for l in range(self.ellmax+1):
        #    for m in range(l+1):
        #        j = hp.sphtfunc.Alm.getidx(self.ellmax, l, m)

        for i in range(self.len):
            kmap_out += weight[i] * self.kmaps[i]

        return kmap_out, weight


class ILC_deproj_CIB():
    
    def __init__(self, deltal, spectra, kmaps, lmax, freqs, fsky):
        """
        Same as standard_ILC, but deprojecting CIB signal.
        
        Be sure to check (maybe need modify) the CIB spectral response in spectral.py.
        """

        self.deltal = deltal
        self.spectra = spectra
        self.kmaps = kmaps
        self.shape = kmaps[0].shape
        self.wcs = kmaps[0].wcs
        self.ellmax = lmax
        self.ells = np.arange(lmax+1)
        self.freqs = freqs
        self.len = len(self.freqs)
        self.fsky = fsky
        self.Clij = np.zeros((self.len, self.len, self.shape[0], self.shape[1]))
        self.epsilon = 1e-10
    
    def get_Cl(self):
        #Clij = np.zeros((self.len, self.len, self.lmax+1))
        for i in range(self.len):
            for j in range(self.len):
                pre_Cl = np.real(self.kmaps[i] * np.conj(self.kmaps[j]))
                self.Clij[i][j] = pre_Cl

        return self.Clij
    
    def isotropic_2d_from_Cl(self, ell, Cl, fill_value="extrapolate"):
        Lx, Ly  = enmap.lmap(self.shape, self.wcs)
        Lmod    = np.hypot(Lx, Ly)

        # 用线性插值把 1D C_ell 映射到每个像素的 |l|
        f = interp1d(ell, Cl, kind="linear", bounds_error=False, fill_value=fill_value)
        S2d = f(Lmod)
        return enmap.enmap(S2d, self.wcs)

    def Smooth_power_2D(self, kpower, delta_ell):
        modlmap = enmap.modlmap(self.shape, self.wcs)
        bin_edges = np.arange(2, self.ellmax, self.deltal)
        binner = s.bin2D(modlmap,bin_edges)
        
        cents, nl1d_real = binner.bin(np.real(kpower))
        S2d = self.isotropic_2d_from_Cl(cents, nl1d_real)

        return S2d
    
    def get_Rlij(self):
        self.get_Cl()
        self.Rlij = np.zeros_like(self.Clij)
        for i in range(len(self.freqs)):
            for j in range(len(self.freqs)):
                self.Rlij[i][j] = self.Smooth_power_2D(self.Clij[i][j], self.deltal)
        return self.Rlij
    
    def get_Rlij_inv(self): 
        try:
            self.Rlij_inv = invert_covariance(self.Rlij, epsilon=0)
        except np.linalg.LinAlgError:
            print("Matrix is singular, considering normalization with epsilon=1e-10.")
            self.Rlij_inv = invert_covariance(self.Rlij, epsilon=1e-10)
        return self.Rlij_inv #index as Rlij_inv[l][i][j]
    
    def get_ab(self):
        #get spectral response vectors
        self.a = np.ones(len(self.freqs)) #index as a[i]
        self.b = self.spectra.cib_spectral_response(self.freqs) #index as b[i]
        return self.a, self.b
    
    def get_ABD(self):
        self.A = np.einsum('ijpx,i,j->px', self.Rlij_inv, self.a, self.a)
        self.B = np.einsum('ijpx,i,j->px', self.Rlij_inv, self.b, self.b)
        self.D = np.einsum('ijpx,i,j->px', self.Rlij_inv, self.a, self.b)
        return self.A, self.B, self.D
    
    def weights(self):
        #get weights
        numerator = np.einsum('ijpx,px,i->jpx', self.Rlij_inv, self.B, self.a) \
                    - np.einsum('ijpx,px,i->jpx', self.Rlij_inv, self.D, self.b)
        denominator = np.einsum('px,px->px', self.A, self.B) - np.einsum('px,px->px', self.D, self.D)
        self.w = numerator/denominator #index as w[i][l]
        print(print('constraint:', np.dot(self.w[:, 10, 10],self.a), np.dot(self.w[:, 10, 10],self.b)))
        return self.w
    
    def do_CILC(self):
        self.get_Rlij()
        self.get_Rlij_inv()
        self.get_ab()
        self.get_ABD()
        self.weights()
        weight = self.w / np.sum(self.w, axis=0)   # I think we need to confirm the normalization here again: \Sum_i w_i = 1  !!!!

        kmap_out = enmap.zeros(self.shape, self.wcs, dtype=np.complex128)
        #for l in range(self.ellmax+1):
        #    for m in range(l+1):
        #        j = hp.sphtfunc.Alm.getidx(self.ellmax, l, m)

        for i in range(self.len):
            kmap_out += weight[i] * self.kmaps[i]

        return kmap_out, weight


class ILC_deproj_tSZ_and_CIB():
    
    def __init__(self, deltal, spectra, kmaps, lmax, freqs, fsky):
        """
        Same as standard_ILC, but deprojecting both tSZ and CIB signals.

        Be sure to check the spectral responses in spectral.py.
        """

        self.deltal = deltal
        self.spectra = spectra
        self.kmaps = kmaps
        self.shape = kmaps[0].shape
        self.wcs = kmaps[0].wcs
        self.ellmax = lmax
        self.ells = np.arange(lmax+1)
        self.freqs = freqs
        self.len = len(self.freqs)
        self.fsky = fsky
        self.Clij = np.zeros((self.len, self.len, self.shape[0], self.shape[1]))
        self.epsilon = 1e-10
    
    def get_Cl(self):
        #Clij = np.zeros((self.len, self.len, self.lmax+1))
        for i in range(self.len):
            for j in range(self.len):
                pre_Cl = np.real(self.kmaps[i] * np.conj(self.kmaps[j]))
                self.Clij[i][j] = pre_Cl

        return self.Clij
    
    def isotropic_2d_from_Cl(self, ell, Cl, fill_value="extrapolate"):
        Lx, Ly  = enmap.lmap(self.shape, self.wcs)
        Lmod    = np.hypot(Lx, Ly)

        # 用线性插值把 1D C_ell 映射到每个像素的 |l|
        f = interp1d(ell, Cl, kind="linear", bounds_error=False, fill_value=fill_value)
        S2d = f(Lmod)
        return enmap.enmap(S2d, self.wcs)

    def Smooth_power_2D(self, kpower, delta_ell):
        modlmap = enmap.modlmap(self.shape, self.wcs)
        bin_edges = np.arange(2, self.ellmax, self.deltal)
        binner = s.bin2D(modlmap,bin_edges)
        
        cents, nl1d_real = binner.bin(np.real(kpower))
        S2d = self.isotropic_2d_from_Cl(cents, nl1d_real)

        return S2d
    
    def get_Rlij(self):
        self.get_Cl()
        self.Rlij = np.zeros_like(self.Clij)
        for i in range(len(self.freqs)):
            for j in range(len(self.freqs)):
                self.Rlij[i][j] = self.Smooth_power_2D(self.Clij[i][j], self.deltal)
        return self.Rlij
    
    def get_Rlij_inv(self): 
        try:
            self.Rlij_inv = invert_covariance(self.Rlij, epsilon=0)
        except np.linalg.LinAlgError:
            print("Matrix is singular, considering normalization with epsilon=1e-10.")
            self.Rlij_inv = invert_covariance(self.Rlij, epsilon=1e-10)
        return self.Rlij_inv #index as Rlij_inv[l][i][j]
    
    def get_abc(self):
        #get spectral response vectors
        self.a = np.ones(len(self.freqs)) #index as a[i]
        self.b = self.spectra.tsz_spectral_response(self.freqs) #index as b[i]
        self.c = self.spectra.cib_spectral_response(self.freqs) #index as c[i]
        return self.a, self.b, self.c
    
    def get_ABCDEFQ(self):
        self.A = np.einsum('ijpx,i,j->px', self.Rlij_inv, self.a, self.a)
        self.B = np.einsum('ijpx,i,j->px', self.Rlij_inv, self.b, self.b)
        self.C = np.einsum('ijpx,i,j->px', self.Rlij_inv, self.c, self.c)
        self.D = np.einsum('ijpx,i,j->px', self.Rlij_inv, self.a, self.b)
        self.E = np.einsum('ijpx,i,j->px', self.Rlij_inv, self.a, self.c)
        self.F = np.einsum('ijpx,i,j->px', self.Rlij_inv, self.b, self.c)
        self.Q = np.einsum('px,px,px->px', self.A, self.B, self.C) + 2*np.einsum('px,px,px->px', self.D, self.E, self.F) \
            - np.einsum('px,px,px->px', self.A, self.F, self.F) - np.einsum('px,px,px->px', self.B, self.E, self.E) \
            - np.einsum('px,px,px->px', self.C, self.D, self.D)
        self.Q[0] = 1
        return self.A, self.B, self.C, self.D, self.E, self.F, self.Q
    
    def weights(self):
        #Define weights
        self.w = np.einsum('ijpx,px,px,px,i->jpx', self.Rlij_inv, 1/self.Q, self.B, self.C, self.a) \
            - np.einsum('ijpx,px,px,px,i->jpx', self.Rlij_inv, 1/self.Q, self.F, self.F, self.a) \
            + np.einsum('ijpx,px,px,px,i->jpx', self.Rlij_inv, 1/self.Q, self.E, self.F, self.b) \
            - np.einsum('ijpx,px,px,px,i->jpx', self.Rlij_inv, 1/self.Q, self.C, self.D, self.b) \
            + np.einsum('ijpx,px,px,px,i->jpx', self.Rlij_inv, 1/self.Q, self.D, self.F, self.c) \
            - np.einsum('ijpx,px,px,px,i->jpx', self.Rlij_inv, 1/self.Q, self.B, self.E, self.c)
        print(print('constraint:', np.dot(self.w[:, 10, 10],self.a), np.dot(self.w[:, 10, 10],self.b), np.dot(self.w[:, 10, 10],self.c)))
        return self.w
    
    def do_CILC(self):
        self.get_Rlij()
        self.get_Rlij_inv()
        self.get_abc()
        self.get_ABCDEFQ()
        self.weights()
        weight = self.w / np.sum(self.w, axis=0)   # I think we need to confirm the normalization here again: \Sum_i w_i = 1  !!!!

        kmap_out = enmap.zeros(self.shape, self.wcs, dtype=np.complex128)
        #for l in range(self.ellmax+1):
        #    for m in range(l+1):
        #        j = hp.sphtfunc.Alm.getidx(self.ellmax, l, m)

        for i in range(self.len):
            kmap_out += weight[i] * self.kmaps[i]

        return kmap_out, weight


