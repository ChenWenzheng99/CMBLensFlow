import healpy as hp
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy import units as u
from astropy.cosmology import Planck15
from scipy.interpolate import griddata

from spectral import *



def PILC(maps, spectral, freqs, deproj=None, weights_read=None):
    """
    Implementation of the internal linear combination (ILC) algorithm on the full sky. 
    The covariance matrix is computed from the whole input maps.
    Deprojection of tSZ, CIB, or both can be performed. Both deprojections in piexl space behave worse than in harmonic space.
    """
    #maps: input multi-frequency maps; np.array with dimensions (n_freqs, n_pixels)
    #fgds: input foreground maps; np.array with dimensions (n_freqs, n_pixels)
    #noise: input noise maps: np.array with dimensions (n_frews, n_pixels)

    if weights_read is None:
        #Computing the covariance of input maps
        cov= np.cov(maps) #numpy function to calculate covariance

        #Computing the inverse of the covariance
        inv_cov = np.linalg.inv(cov) #function to calculate the inv of a square matrix

        #Component spectral energy density
        a = np.ones(maps.shape[0]) 
        if deproj == 'tSZ':
            b = spectral.tsz_spectral_response(freqs) #index as b[i]
        elif deproj == 'CIB':
            b = spectral.cib_spectral_response(freqs)
        elif deproj == 'joint':
            b = spectral.tsz_spectral_response(freqs) #index as b[i]
            c = spectral.cib_spectral_response(freqs) #index as c[i]

        #Computing the weights
        if deproj == None or deproj == 'standard':
            w=(a @ inv_cov /(a @ (inv_cov @a)))
            w= np.sum(inv_cov, axis =1)/np.sum(inv_cov) #sums elements in an array. axis =1 sums along rows. None= sum of all elements
        elif deproj == 'tSZ' or deproj == 'CIB':
            A = np.einsum('ij,i,j->', inv_cov, a, a)
            B = np.einsum('ij,i,j->', inv_cov, b, b)
            D = np.einsum('ij,i,j->', inv_cov, a, b)
            numerator = np.einsum('ij, ,j->j', inv_cov, B, a) - np.einsum('ij, ,j->j', inv_cov, D, b)
            denominator = A*B - D*D
            w = numerator/denominator
        elif deproj == 'joint':
            A = np.einsum('ij,i,j->', inv_cov, a, a)
            B = np.einsum('ij,i,j->', inv_cov, b, b)
            C = np.einsum('ij,i,j->', inv_cov, c, c)
            D = np.einsum('ij,i,j->', inv_cov, a, b)
            E = np.einsum('ij,i,j->', inv_cov, a, c)
            F = np.einsum('ij,i,j->', inv_cov, b, c)
            Q = A*B*C + 2*D*E*F - A*F*F - B*E*E - C*D*D
            
            w = np.einsum('ij,i->j', inv_cov, 1/Q * (B*C - F*F) * a + 1/Q * (E*F - C*D) * b + 1/Q * (D*F - B*E) * c)
    else:
        w = weights_read[i]

        #Linearly combining the maps with the corresponding weights to obtain the output ILC solution for the maps, foregrounds, and noise
        out = np.einsum ('j, ji->i', w, maps) 


    #Einstein Summation notation. vector of dim j multiplied by vector of dim i to get a vector of dim i

    return out, w



def PILC_patched(maps, nside, nside_patches, mask_in, spectral, freqs, deproj=None, weights_read=None):
    """
    Implementation of the internal linear combination (ILC) algorithm on the full sky. 
    The covariance matrix is computed from patches of the input maps.
    Deprojection of tSZ, CIB, or both can be performed. Both deprojections in piexl space behave worse than in harmonic space.
    """
    #maps: input multi-frequency maps; np.array with dimensions (n_freqs, n_pixels)
    #fgds: input foreground maps; np.array with dimensions (n_freqs, n_pixels)
    #noise: input noise maps: np.array with dimensions (n_frews, n_pixels)

    #Component spectral energy density
    a = np.ones(maps.shape[0]) 
    if deproj == 'tSZ':
        b = spectral.tsz_spectral_response(freqs) #index as b[i]
    elif deproj == 'CIB':
        b = spectral.cib_spectral_response(freqs)
    elif deproj == 'joint':
        b = spectral.tsz_spectral_response(freqs) #index as b[i]
        c = spectral.cib_spectral_response(freqs) #index as c[i]


    patches = hp.ud_grade(np.arange(hp.nside2npix(nside_patches)), nside)
    Npatch = hp.nside2npix(nside_patches)

    outmap = np.zeros(hp.nside2npix(nside))

    masks = []
    fields = []
    inv_covs = []
    weights = []
    for i in range(Npatch):
        masks.append((patches == i).astype(int))
        fields.append(maps * masks[i])

    for i in range(Npatch):
        if weights_read is None:
            neighbor_pix = hp.get_all_neighbours(nside_patches, i, phi=None, nest=False, lonlat=False)
            fsky_i = np.sum(masks[i]*mask_in + np.sum([masks[j]*mask_in for j in neighbor_pix], axis=0)) / np.sum(mask_in)  #sums elements in an array. axis =1 sums along rows. None= sum of all elements
            field_i = fields[i] + np.sum([fields[j] for j in neighbor_pix], axis=0)
            inv_covs.append(np.linalg.inv(np.cov(field_i)) / fsky_i) #numpy function to calculate covariance 

            #Computing the weights
            if deproj == None or deproj == 'standard':
                w=(a @ inv_covs[i] /(a @ (inv_covs[i] @a)))
                #w= np.sum(inv_covs[i], axis =1)/np.sum(inv_covs) #sums elements in an array. axis =1 sums along rows. None= sum of all elements
            elif deproj == 'tSZ' or deproj == 'CIB':
                A = np.einsum('ij,i,j->', inv_covs[i], a, a)
                B = np.einsum('ij,i,j->', inv_covs[i], b, b)
                D = np.einsum('ij,i,j->', inv_covs[i], a, b)
                numerator = np.einsum('ij, ,j->j', inv_covs[i], B, a) - np.einsum('ij, ,j->j', inv_covs[i], D, b)
                denominator = A*B - D*D
                w = numerator/denominator
            elif deproj == 'joint':
                A = np.einsum('ij,i,j->', inv_covs[i], a, a)
                B = np.einsum('ij,i,j->', inv_covs[i], b, b)
                C = np.einsum('ij,i,j->', inv_covs[i], c, c)
                D = np.einsum('ij,i,j->', inv_covs[i], a, b)
                E = np.einsum('ij,i,j->', inv_covs[i], a, c)
                F = np.einsum('ij,i,j->', inv_covs[i], b, c)
                Q = A*B*C + 2*D*E*F - A*F*F - B*E*E - C*D*D
                w = np.einsum('ij,i->j', inv_covs[i], 1/Q * (B*C - F*F) * a + 1/Q * (E*F - C*D) * b + 1/Q * (D*F - B*E) * c)
            
            #w = w / np.sum(w)

            outmap += np.einsum ('j, ji->i', w, fields[i]) 
            weights.append(w)
        else:
            w = weights_read[i]
            outmap += np.einsum ('j, ji->i', w, fields[i]) 
            weights.append(w)

    #Einstein Summation notation. vector of dim j multiplied by vector of dim i to get a vector of dim i

    return outmap, weights


def PILC_patched2(maps, nside, nside_patches, mask_in, spectral, freqs, deproj=None, weights_read=None):
    """
    Implementation of the internal linear combination (ILC) algorithm on the full sky. 
    The covariance matrix is computed from patches of the input maps.
    Deprojection of tSZ, CIB, or both can be performed. Both deprojections in piexl space behave worse than in harmonic space.
    """
    #maps: input multi-frequency maps; np.array with dimensions (n_freqs, n_pixels)
    #fgds: input foreground maps; np.array with dimensions (n_freqs, n_pixels)
    #noise: input noise maps: np.array with dimensions (n_frews, n_pixels)

    epsilon = 1e-15

    #Component spectral energy density
    a = np.ones(maps.shape[0]) 
    if deproj == 'tSZ':
        b = spectral.tsz_spectral_response(freqs) #index as b[i]
    elif deproj == 'CIB':
        b = spectral.cib_spectral_response(freqs)
    elif deproj == 'joint':
        b = spectral.tsz_spectral_response(freqs) #index as b[i]
        c = spectral.cib_spectral_response(freqs) #index as c[i]


    patches = hp.ud_grade(np.arange(hp.nside2npix(nside_patches)), nside)
    Npatch = hp.nside2npix(nside_patches)

    outmap = np.zeros(hp.nside2npix(nside))

    weights = []
    for i in range(Npatch):
        masks = np.zeros(hp.nside2npix(nside))

        mask_i = (patches == i).astype(int)

        if weights_read is None:
            neighbor_pix = hp.get_all_neighbours(nside_patches, i, phi=None, nest=False, lonlat=False)
            for j in neighbor_pix:
                masks += ((patches == j).astype(int))

            fsky_i = np.sum(masks*mask_in , axis=0) / np.sum(mask_in)  #sums elements in an array. axis =1 sums along rows. None= sum of all elements
            fields = maps * masks
            inv_covs = (np.linalg.inv(np.cov(fields) + epsilon * np.eye(np.cov(fields).shape[0])) / fsky_i) #numpy function to calculate covariance 

            #Computing the weights
            if deproj == None or deproj == 'standard':
                #w=(a @ inv_covs /(a @ (inv_covs @a)))
                w = np.einsum('i,ij->j', a, inv_covs) / np.einsum('i,ij,j->', a, inv_covs, a)
                #w= np.sum(inv_covs, axis =1)/np.sum(inv_covs) #sums elements in an array. axis =1 sums along rows. None= sum of all elements
            elif deproj == 'tSZ' or deproj == 'CIB':
                A = np.einsum('ij,i,j->', inv_covs, a, a)
                B = np.einsum('ij,i,j->', inv_covs, b, b)
                D = np.einsum('ij,i,j->', inv_covs, a, b)
                numerator = np.einsum('ij, ,j->j', inv_covs, B, a) - np.einsum('ij, ,j->j', inv_covs, D, b)
                denominator = A*B - D*D
                w = numerator/denominator
            elif deproj == 'joint':
                A = np.einsum('ij,i,j->', inv_covs, a, a)
                B = np.einsum('ij,i,j->', inv_covs, b, b)
                C = np.einsum('ij,i,j->', inv_covs, c, c)
                D = np.einsum('ij,i,j->', inv_covs, a, b)
                E = np.einsum('ij,i,j->', inv_covs, a, c)
                F = np.einsum('ij,i,j->', inv_covs, b, c)
                Q = A*B*C + 2*D*E*F - A*F*F - B*E*E - C*D*D
                w = np.einsum('ij,i->j', inv_covs, 1/Q * (B*C - F*F) * a + 1/Q * (E*F - C*D) * b + 1/Q * (D*F - B*E) * c)
            
            outmap += np.einsum ('j, ji->i', w, mask_i * maps) 
            weights.append(w)
        else:
            w = weights_read[i]
            outmap += np.einsum ('j, ji->i', w, mask_i * maps) 
            weights.append(w)

    return outmap, weights