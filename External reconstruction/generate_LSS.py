import numpy as np
import healpy as hp

"""
Based on the theoretical auto- and cross- power spectra of these tracers(e.g. CMB,CIB,LSST), 
we calculate some scale factors A and the noise power, the tracer map is simply the sum of the scaled true convergence map and the noise realization from the calculated power spectrum.
"""

def shorten_alm(input_alm, lmax_new):
    """
    Used for cut alm.
    """
    lmax_old = hp.Alm.getlmax(len(input_alm))
    new_size = hp.Alm.getsize(lmax_new)
    output_alm = np.zeros(new_size, dtype=np.complex)
    
    index_in_new = np.arange(len(output_alm))
    l, m = hp.Alm.getlm(lmax_new, i=index_in_new)
    output_alm[index_in_new] = input_alm[hp.Alm.getidx(lmax_old, l, m)]
    return output_alm

def read_phi_alms(lmax, phi_alm_dir):
    phi_alm = hp.read_alm(phi_alm_dir)
    return shorten_alm(phi_alm, lmax)

def pad_cls(orig_cl, ells, lmin, lmax):
    cl_padded = np.zeros(ells.shape)
    cl_padded[lmin:lmax+1] = orig_cl
    return cl_padded

def corrcoeff(cross, auto1, auto2):
    """
    Used for calculate rho.
    """
    return cross/np.sqrt(auto1*auto2)

def calculate_sim_weights(cl):
    '''
    Calculate the weights A_l^{ij} 
    and 
    the auxiliary spectra C_l^{ij}={C_l^{uu}, C_l^{ee}, ...} from which to draw the alm coefficients a_p={u_{lm}, e_{lm}, ...}

    Parameters:
    cl : ndarray
        Input : auto- and cross- power spectra of each tracers with shape (num_of_tracers, num_of_tracers, num_of_multipoles)

    Returns:
    aux_cl : ndarray
        Auxiliary spectra with shape (num_of_tracers, num_of_multipoles)
    A : ndarray
        Weights for the alms with shape (num_of_tracers, num_of_tracers, num_of_multipoles)
    '''
    num_of_tracers = len(cl[:, 0, 0])
    num_of_multipoles = len(cl[0, 0, :])
    aux_cl = np.zeros((num_of_tracers, num_of_multipoles), dtype='complex128')  # Auxiliary spectra
    A = np.zeros((num_of_tracers, num_of_tracers, num_of_multipoles), dtype='complex128')  # Weights for the alms

    for j in range(num_of_tracers):
        for i in range(num_of_tracers):
            if j > i:
                continue
            else:
                aux_cl[j, :] = np.nan_to_num(cl[j, j, :])
                for p in range(j):
                    aux_cl[j] -= np.nan_to_num(A[j, p, :]**2 * aux_cl[p, :])

                A[i, j, :] = np.nan_to_num((1. / aux_cl[j, :]) * cl[i, j, :])

                for p in range(j):
                    A[i, j, :] -= np.nan_to_num((1. / aux_cl[j, :]) * A[j, p, :] * A[i, p, :] * aux_cl[p, :])
                    
    return aux_cl, A

def draw_gaussian_a_p(input_kappa_alm, aux_cl,):
    '''
    Draw a_p alms from distributions with the right auxiliary spectra.
    '''
    num_of_tracers = len(aux_cl[:,0])
    a_alms = np.zeros((num_of_tracers, len(input_kappa_alm)), dtype='complex128') #Unweighted alm components

    a_alms[0,:] = input_kappa_alm
    for j in range(1, num_of_tracers):
        a_alms[j,:] = hp.synalm(aux_cl[j,:], lmax=len(aux_cl[0,:])-1)

    return a_alms

def generate_individual_gaussian_tracers(a_alms, A, nlkk):
    '''
    Put all the weights and alm components together to give appropriately correlated tracers
    '''
    num_of_tracers = len(a_alms[:,0])
    tracer_alms = np.zeros((num_of_tracers, len(a_alms[0,:])), dtype='complex128') #Appropriately correlated final tracers

    tracer_alms[0,:] = a_alms[0,:] + hp.synalm(nlkk, lmax=len(A[0,0,:])-1)
    for i in range(1,num_of_tracers):
        for j in range(i+1):
            tracer_alms[i,:] += hp.almxfl(a_alms[j,:], A[i,j,:])

    return tracer_alms

def ones(nside, lmax, pmap, nlpp, cls, wantmap=None):
    """
    nside:
    lmax:
    pmap: Input lenisng potential map (signal only)
    nlpp: Reconstruction noise power spectrum, use the theory one. If multi-frequency reconstruction were used, this should be their inverse-combination noise power.
    cls:  Power spectrum matrix in the shapes of (num_of_tracers, num_of_tracers, num_of_multipoles), for example 3 tracers (k, i, g):
          the cls will be in the shapes of (3,3,lmax), and cls[i,j,:] should be the power spcreum of tracer i and tracer j.

    Returns:
    tracer_alms: ndarray
        The final tracers in the shape of (num_of_tracers)
    """
    ls = np.arange(lmax+1)

    plm = hp.map2alm(pmap, lmax)
    q2k = lambda l: l*(l + 1) / 2
    klm = hp.almxfl(plm, q2k(ls))

    num_of_tracers = len(cls[:, 0, 0])

    # Firstly, get the noise power and the weights of each tracers 
    cls_noises, Als = calculate_sim_weights(cls)

    # Draw the raw alm components
    alm_raw = draw_gaussian_a_p(klm, cls_noises)

    # Generate the final tracers
    nlkk = nlpp * (ls*(ls+1))**2 / 4
    alm_tracers = generate_individual_gaussian_tracers(alm_raw, Als, nlkk)
    alm_tracers[:, 0] = 0   # Remove monopole

    if wantmap is not None:
        tracer_maps = []
        for i in range(num_of_tracers):
            tracer_maps.append(hp.alm2map(alm_tracers[i], nside))
        return tracer_maps
    else:
        return alm_tracers
    