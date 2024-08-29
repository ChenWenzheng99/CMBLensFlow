import numpy as np
import curvedsky
from utils_mine import *

def lensingb_mine(lmax, elmin, elmax, plmin, plmax, wElm, wplm, nside_t=0, gtype='p',sm_mask=None):
    """
    Warning: This function automatically do the E2B correction, which is necessary for partial sky. If you want to use this function for full sky, please set bi_mask = None.
    The output B template map had better been cut when calculating the pseudo-Cl, cause the edge of the map is not guaranteed to be accurate.
    """
    nside = nside_t if nside_t != 0 else 2**(int(np.log2(max(elmax, plmax))))
    npix = 12*nside**2

    l_cut_e = elmax
    l_cut_p = plmax
    elmax = 3*nside-1
    plmax = 3*nside-1

    if sm_mask is None:
        sm_mask = np.ones(npix, dtype=float)

    ilk = np.ones(plmax + 1, dtype=float)
    if gtype == 'k':
        for l in range(1, plmax + 1):
            ilk[l] = 2.0 / (l * (l + 1))

    #A1 = np.zeros((npix, 2), dtype=float)
    #A3 = np.zeros((npix, 2), dtype=float)
    #A = np.zeros((npix, 2), dtype=float)
            
    alm = np.zeros((2, elmax + 1, elmax + 1), dtype=complex)
    for l in range(elmin, elmax + 1):
        alm[1, l, :] = wElm[l, :] * np.sqrt((l + 2) * (l - 1) * 0.5)
        #alm[0, 0, :] = 0
    if elmin > 0:
        alm[1, :elmin, :] = 0
    alm[1,l_cut_e+1:,:] = 0
    A1_0,A1_1 = curvedsky.utils.hp_alm2map_spin(nside, elmax, elmax, 1, alm[1, :elmax + 1, :], alm[1, :elmax + 1, :]*0) 
    A1 = np.column_stack((A1_0, A1_1))

    alm = np.zeros((2, elmax + 1, elmax + 1), dtype=complex)
    for l in range(elmin, elmax + 1):
        alm[1, l, :] = wElm[l, :] * np.sqrt((l - 2) * (l + 3) * 0.5)
        #alm[0, 0, :] = 0
    if elmin > 0:
        alm[1, :elmin, :] = 0
    alm[1,l_cut_e+1:,:] = 0
    A3_0,A3_1 = curvedsky.utils.hp_alm2map_spin(nside, elmax, elmax, 3, alm[1, :elmax + 1, :], alm[1, :elmax + 1, :]*0) 
    A3 = np.column_stack((A3_0, A3_1))

    alm = np.zeros((2, plmax + 1, plmax + 1), dtype=complex)
    for l in range(plmin, plmax + 1):
        alm[1, l, :] = wplm[l, :] * np.sqrt(l * (l + 1) * 0.5) * ilk[l]  
        #alm[0, 0, :] = 0
    if plmin > 0:
        alm[1, :plmin, :] = 0
    alm[1,l_cut_p+1:,:] = 0
    A_0,A_1 = curvedsky.utils.hp_alm2map_spin(nside, plmax, plmax, 1, alm[1, :plmax + 1, :], alm[1, :plmax + 1, :]*0) 
    A = np.column_stack((A_0, A_1))

    #map = A1*A - A3*conjg(A)                 我的推导差个负号，其他一致，待检查，程序是对的                       
    #    = (rA1+iu*iA1)*(rA+iu*iA) - (rA3+iu*iA3)*(rA-iu*iA)
    #    = rA*(rA1-rA3+iu*(iA1-iA3)) + iA*(iu*rA1-iA1+iu*rA3-iA3)
    map_real = A[:, 0] * (A1[:, 0] - A3[:, 0]) - A[:, 1] * (A1[:, 1] + A3[:, 1])
    map_imag = A[:, 0] * (A1[:, 1] - A3[:, 1]) + A[:, 1] * (A1[:, 0] + A3[:, 0])

    map = np.column_stack((map_real, map_imag)) 

    #alm = np.zeros((2, lmax + 1, lmax + 1), dtype=complex)
    #alm[2, :, :] = ifft2(map).reshape((lmax + 1, lmax + 1))

    #lBlm = alm[2, :, :]
    

    map_b,l_bin,dlbb_bin = E2Bcorrection(nside,50,[map[:,0]*sm_mask,map[:,0]*sm_mask,map[:,1]*sm_mask],sm_mask,lmax=3000,flag='clean',n_iter=3, is_Dell=True)

    return map_b,map