import numpy as np
import healpy as hp
#-------------------------------------------------------------------------------------------------
# See /root/Testarea/prototype/A_PARTIAL_SKY/test_antilensing.ipynb for example and performance
#-------------------------------------------------------------------------------------------------

def grad1(alm,nside,lmax=None,mmax=None):
    m, d_theta, d_phi = hp.sphtfunc.alm2map_der1(alm, nside, lmax=lmax, mmax=mmax)
    d1m = np.stack((d_theta, d_phi), axis=1)
    return m, d1m

def grad2(alm,nside,lmax=None,mmax=None):
    m, d_theta, d_phi = hp.sphtfunc.alm2map_der1(alm, nside, lmax=None, mmax=None)
    alm_dt = hp.map2alm(d_theta, lmax=lmax, mmax=mmax)
    alm_dp = hp.map2alm(d_phi, lmax=lmax, mmax=mmax)
    m_dt, d_theta_theta, d_theta_phi = hp.sphtfunc.alm2map_der1(alm_dt, nside, lmax=None, mmax=None)
    m_dp, d_phi_theta, d_phi_phi = hp.sphtfunc.alm2map_der1(alm_dp, nside, lmax=None, mmax=None)

    d1m = np.stack((d_theta, d_phi), axis=1)
    d2m = np.stack((d_theta_theta, d_theta_phi, d_phi_phi), axis=1)

    return m, d1m, d2m

def beta_fast_NEW(nside,alm,nremap=5):
    """
    NOTICE: The defualt parameters are set to lead the most close results comparing to cmblenplus .(After test, 1st der step = 1e-13, 2nd der step = 1e-50 and nremap=1 lead to the best convergence)
    """
    m,alpha,dalpha = grad2(alm, nside, lmax=None, mmax=None)
    beta = antilensing_angle(alpha, dalpha, nremap=nremap)
    return beta


def numerical_gradient1(map_values, nside, delta_theta=1e-13, delta_phi=1e-13):
    """
    Get the numerical 1st derivative of a map directly, without using the spherical harmonic transform. der1 = dT/d(theta), dT/d(phi)/sin(theta)
    
    map_values: please input only one healpix map
    nside: the nside of the map

    NOTICE: 1.the map input had better be a partial sky map, otherwise the calculation may be very slow when nside is large
            2.the step size of delta_theta and delta_phi had better be small enough, this could be a proper value: 1e-13 (this is settled by comparing the result of curvedsky.delens.phi2grad(nside, lmax, plm))
    """
    npix = hp.nside2npix(nside)
    der1_theta = np.zeros(npix)
    der1_phi = np.zeros(npix)
    
    theta, phi = hp.pix2ang(nside, np.arange(npix))

    # 预先计算 sin(theta)
    sin_theta = np.sin(theta)

    for i in range(npix):
        # 如果当前像素的函数值为零，直接跳过
        if map_values[i] == 0:
            continue

        # 计算θ方向上的数值梯度（一阶导数）
        der1_theta[i] = (hp.get_interp_val(map_values, theta[i] + delta_theta/2, phi[i]) - 
                         hp.get_interp_val(map_values, theta[i] - delta_theta/2, phi[i])) / delta_theta

        # 计算φ方向上的数值梯度（一阶导数）
        der1_phi[i] = (hp.get_interp_val(map_values, theta[i], phi[i] + delta_phi/2) - 
                       hp.get_interp_val(map_values, theta[i], phi[i] - delta_phi/2)) / (delta_phi * sin_theta[i])
    alpha = np.stack((der1_theta, der1_phi), axis=1)
        
    return alpha


def numerical_gradient2(map_values, nside, delta_theta=1e-5, delta_phi=1e-5):
    """
    Get the numerical 2nd of a map derivative of a map directly, without using the spherical harmonic transform. der2 = d^2T/d(theta^2), d^2T/d(theta)/d(phi)/sin(theta), d^2T/d(phi^2)/sin(theta)^2

    map_values: please input only one healpix map
    nside: the nside of the map

    NOTICE: 1.the map input had better be a partial sky map, otherwise the calculation may be very slow when nside is large
            2.the step size of delta_theta and delta_phi had better be small enough, this could be a proper value: 1e-5 or 1e-50
            (this doesn't matter, because the result is not sensitive to the step size, and when the 2nd derivative is used to calculate antilensing angle, 
            the step size seem to have no effect on the final result, a 1e-50 step only lead to convergence after one recursion, 

            (this is settled by comparing the result of curvedsky.delens.shiftvec(nside, lmax, plm, nremap))

    """
    npix = hp.nside2npix(nside)
    der2_theta_theta = np.zeros(npix)
    der2_theta_phi = np.zeros(npix)
    der2_phi_phi = np.zeros(npix)

    theta, phi = hp.pix2ang(nside, np.arange(npix))

    # 预先计算 sin(theta)
    sin_theta = np.sin(theta)

    for i in range(npix):

        # 如果当前像素的函数值为零，直接跳过
        if map_values[i] == 0:
            continue

        # Calculate the numerical gradient (second derivative) in theta direction
        der2_theta_theta[i] = (hp.get_interp_val(map_values, theta[i] + delta_theta, phi[i] ) - 
                               2 * hp.get_interp_val(map_values, theta[i], phi[i] ) + 
                               hp.get_interp_val(map_values, theta[i] - delta_theta, phi[i] )) / (delta_theta ** 2)


        # Calculate the numerical gradient (second derivative) in theta-phi direction
        der2_theta_phi[i] = (hp.get_interp_val(map_values, theta[i] + delta_theta/2, phi[i] + delta_phi/2) - 
                             hp.get_interp_val(map_values, theta[i] - delta_theta/2, phi[i] + delta_phi/2) - 
                             hp.get_interp_val(map_values, theta[i] + delta_theta/2, phi[i] - delta_phi/2) + 
                             hp.get_interp_val(map_values, theta[i] - delta_theta/2, phi[i] - delta_phi/2)) / (delta_theta * delta_phi * sin_theta[i])
        
        # Calculate the numerical gradient (second derivative) in phi direction
        der2_phi_phi[i] = (hp.get_interp_val(map_values, theta[i], phi[i] + delta_phi) - 
                           2 * hp.get_interp_val(map_values, theta[i], phi[i]) + 
                           hp.get_interp_val(map_values, theta[i], phi[i] - delta_phi)) / (delta_phi ** 2 * sin_theta[i]**2)
    dalpha = np.stack((der2_theta_theta, der2_theta_phi, der2_phi_phi), axis=1)
    return dalpha

def antilensing_angle(alpha, dalpha, nremap=1):
    """
    Calculate the antilensing angle beta from the lensing angle alpha and its derivative dalpha directly, without using the spherical harmonic transform.

    alpha: the lensing angle, a 2D array with shape (npix, 2)
    dalpha: the derivative of the lensing angle, a 2D array with shape (npix, 3)
    nremap: the number of recursion

    NOTICE: When 2nd der step is 1e-50 (or 1e-30,1e-40 etc.), nremap=1 leads to the best convergence.(This also the best perfornmance). Or you can choose 2nd der step = 1e-5 and nremap=1, which leads to a so-so performance.
            2nd der step > 1e-5 &  1e-8 < 2nd der step  <1e-20 lead to a bad performance.
    """
    npix = alpha.shape[0]
    beta = np.zeros((npix, 2))

    for _ in range(nremap):
        beta[:, 0] = alpha[:, 0] - dalpha[:, 0] * beta[:, 0] - dalpha[:, 1] * beta[:, 1]
        beta[:, 1] = alpha[:, 1] - dalpha[:, 1] * beta[:, 0] - dalpha[:, 2] * beta[:, 1]
    beta = - beta
    return beta


def beta_fast(nside,map,dtheta1=1e-13,dphi1=1e-13,dtheta2=1e-50,dphi2=1e-50,nremap=1):
    """
    NOTICE: The defualt parameters are set to lead the most close results comparing to cmblenplus .(After test, 1st der step = 1e-13, 2nd der step = 1e-50 and nremap=1 lead to the best convergence)
    """
    alpha = numerical_gradient1(map, nside, delta_theta=dtheta1, delta_phi=dphi1)
    dalpha = numerical_gradient2(map, nside, delta_theta=dtheta2, delta_phi=dphi2)
    beta = antilensing_angle(alpha, dalpha, nremap=nremap)
    return beta


def antilensing_angle_test(alpha, dalpha, nremap=3):
    """
    废弃，错误
    """
    npix = alpha.shape[0]
    beta = np.zeros((npix, 2))

    for _ in range(nremap):
        beta[:, 0] = alpha[:, 0] - dalpha[:, 0] * beta[:, 0] - dalpha[:, 1] * beta[:, 1] - dalpha[:, 0] * alpha[:, 0] - dalpha[:, 1] * alpha[:, 1]
        beta[:, 1] = alpha[:, 1] - dalpha[:, 1] * beta[:, 0] - dalpha[:, 2] * beta[:, 1] - dalpha[:, 1] * alpha[:, 0] - dalpha[:, 2] * alpha[:, 1]
    beta = - beta
    return beta

