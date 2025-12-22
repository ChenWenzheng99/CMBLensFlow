import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy import units as u
from astropy.cosmology import Planck15
from scipy.interpolate import griddata


class spectra():
    
    def __init__(self, base_dir, ellmax, freqs, comps, gg_shot_noise, CIB_model='H13', noise=None):
        self.base_dir = base_dir
        self.ellmax = ellmax
        self.freqs = freqs
        self.comps = comps
        self.gg_shot_noise = gg_shot_noise
        self.noise = noise #dimensions (nfreqs x ellmax)
        #index as all_spectra[a][i][b][j] for cross-spectrum of component a at frequency i with component b at frequency j
        self.all_spectra = np.zeros((len(comps), len(freqs), len(comps), len(freqs), ellmax+1)) 
        #index as comp_cross_g_spectra[a][i][b] for cross-spectrum component a at frequency i with g blue green red
        self.comp_cross_g_spectra = np.zeros((len(comps), len(freqs), 3, ellmax+1))
        #g_a x g_b spectra, index 0 for blue, 1 for green, 2 for red
        self.gg_spectrum = np.zeros((3,3,ellmax+1))
        #list of ells
        self.ells = np.arange(ellmax+1)
        self.CIB_model = CIB_model
    
    @staticmethod
    def get_file_name(base_dir, freq1, freq2, comp1, comp2):
        fname = f'{base_dir}/ell_dl_'
        if comp1==comp2=='CMB':
            return f'{base_dir}/ell_dl_CMB_lensed.txt'
        else:
            fname += f'{freq1}x{freq2}_GHz_{comp1}x{comp2}.txt' 
        return fname
    
    def log_interp(self, ells, spectrum):
        if np.all(spectrum >= 0):
            log_spectrum = np.log(spectrum)
            f = interp1d(ells, log_spectrum, fill_value="extrapolate", kind="linear")
            return np.exp(f(self.ells))
        else:
            log_neg_spectrum = np.log(-spectrum)
            f = interp1d(ells, log_neg_spectrum, fill_value="extrapolate", kind="linear")
            return -np.exp(f(self.ells))
    
    def cubic_interp(self, ells, spectrum):
        f = interp1d(ells, spectrum, fill_value="extrapolate", kind='cubic')
        return f(self.ells)
        
    
    
    def get_ksz_auto_spectra(self, plot=False):
        ksz_patchy_file = open(f'{self.base_dir}/FBN_kSZ_PS_patchy.d.txt', 'r')
        rows = ksz_patchy_file.readlines()
        for i, line in enumerate(rows):
            rows[i] = line.lstrip(' ').replace('\n', '').split()
        rows = np.asarray(rows, dtype=np.float32)
        cols = np.transpose(rows)
        ells_ksz_patchy, ksz_patchy = cols
        ksz_patchy = self.cubic_interp(ells_ksz_patchy, ksz_patchy)

        ksz_nonpatchy_file = open(f'{self.base_dir}/FBN_kSZ_PS.d.txt', 'r')
        rows = ksz_nonpatchy_file.readlines()
        for i, line in enumerate(rows):
            rows[i] = line.lstrip(' ').replace('\n', '').split()
        rows = np.asarray(rows, dtype=np.float32)
        cols = np.transpose(rows)
        ells_ksz_nonpatchy, ksz_nonpatchy = cols
        ksz_nonpatchy = self.cubic_interp(ells_ksz_nonpatchy, ksz_nonpatchy)

        ksz = ksz_patchy + ksz_nonpatchy

        if plot:
            plt.clf()
            plt.plot(self.ells, ksz_patchy, label='patchy')
            plt.plot(self.ells, ksz_nonpatchy, label='non-patchy')
            plt.plot(self.ells, ksz, label='total kSZ')
            plt.title('kSZ Auto-Spectra')
            plt.legend()
            plt.xlabel(r'$\ell$')
            plt.ylabel(r'$D_{\ell}$ [$\mu K^2$]')
        ksz = ksz/((self.ells)*(self.ells+1))*(2*np.pi)
        ksz[0] = 0
        return ksz
    
    def get_noise_auto_spectra(self, freq, plot=False):
        planck_noise_channels = {30, 44, 70, 100, 143, 217, 353}
        SO_noise_channels = {93, 145, 225, 280}
        if freq in SO_noise_channels:
            noise_file = open(f'{base_dir}/so_noise/noise_{freq}GHz.txt', 'r')
        else:
            noise_file = open(f'{base_dir}/planck_noise/noise_{freq}GHz.txt', 'r')
        rows = noise_file.readlines()
        for i, line in enumerate(rows):
            rows[i] = line.lstrip(' ').replace('\n', '').split()
        rows = np.asarray(rows, dtype=np.float32)
        ells_noise, noise = rows
        noise = self.cubic_interp(ells_noise, noise)
        if plot:
            plt.plot(self.ells, self.ells*(self.ells+1)*noise/(2*np.pi)) #noise needs to be put into Dl for plotting
            plt.title(f'Noise Auto-Spectrum {freq} GHz')
            plt.xlabel(r'$\ell$')
            plt.ylabel(r'$D_{\ell}$ [$\mu K^2$]')
        return noise
    
    @staticmethod
    def tsz_spectral_response(freqs): #input frequency in GHz
        T_cmb = 2.726
        h = 6.62607004*10**(-34)
        kb = 1.38064852*10**(-23)
        f = 1. #fsky
        response = []
        for freq in freqs:
            x = h*(freq*10**9)/(kb*T_cmb) #x is v/56.9 GHz
            response.append(T_cmb*(x*1/np.tanh(x/2)-4)) #was factor of tcmb microkelvin before
        return np.array(response)
    
    def cib_spectral_response(self, freqs): #input frequency in GHz
        # function from pyilc
        # CIB = modified blackbody here
        # N.B. overall amplitude is not meaningful here; output ILC map (if you tried to preserve this component) would not be in sensible units

        TCMB = 2.726 #Kelvin
        TCMB_uK = 2.726e6 #micro-Kelvin
        hplanck=6.626068e-34 #MKS
        kboltz=1.3806503e-23 #MKS
        clight=299792458.0 #MKS

        # function needed for Planck bandpass integration/conversion following approach in Sec. 3.2 of https://arxiv.org/pdf/1303.5070.pdf
        # blackbody derivative
        # units are 1e-26 Jy/sr/uK_CMB
        def dBnudT(nu_ghz):
            nu = 1.e9*np.asarray(nu_ghz)
            X = hplanck*nu/(kboltz*TCMB)
            return (2.*hplanck*nu**3.)/clight**2. * (np.exp(X))/(np.exp(X)-1.)**2. * X/TCMB_uK

        # conversion from specific intensity to Delta T units (i.e., 1/dBdT|T_CMB)
        #   i.e., from W/m^2/Hz/sr (1e-26 Jy/sr) --> uK_CMB
        #   i.e., you would multiply a map in 1e-26 Jy/sr by this factor to get an output map in uK_CMB
        def ItoDeltaT(nu_ghz):
            return 1./dBnudT(nu_ghz)

        
        if self.CIB_model == 'H13':
            Tdust_CIB = 24.0       #CIB effective dust temperature [K] (Table 9 of http://www.aanda.org/articles/aa/pdf/2014/11/aa22093-13.pdf)
            beta_CIB = 1.2         #CIB modified blackbody spectral index (Table 9 of http://www.aanda.org/articles/aa/pdf/2014/11/aa22093-13.pdf ; Table 10 of that paper contains CIB monopoles)
        elif self.CIB_model == 'P14':
            Tdust_CIB = 20.0       #CIB effective dust temperature [K] (Table 9 of http://www.aanda.org/articles/aa/pdf/2014/11/aa22093-13.pdf)
            beta_CIB = 1.45         #CIB modified blackbody spectral index (Table 9 of http://www.aanda.org/articles/aa/pdf/2014/11/aa22093-13.pdf ; Table 10 of that paper contains CIB monopoles)
        elif self.CIB_model == 'S10_basic':  # N. Sehgal, P. Bode, S. Das, C. Hernand ez-Monteagudo, K. Huffenberger, Y.-T. Lin, J. P. Ostriker, and H. Trac, Astrophys. J. 709, 920 (2010), 0908.0540.
            Tdust_CIB = 25.0
            beta_CIB = 1.4
        elif self.CIB_model == 'S10_high':  # N. Sehgal, P. Bode, S. Das, C. Hernand ez-Monteagudo, K. Huffenberger, Y.-T. Lin, J. P. Ostriker, and H. Trac, Astrophys. J. 709, 920 (2010), 0908.0540.
            Tdust_CIB = 40.0
            beta_CIB = 1.3
        elif self.CIB_model == 'S10_fit':  # fit by hand, see : /root/download/ILC_ALL/ILC_MINE/fit_cib.ipynb
            Tdust_CIB = 7.55
            beta_CIB = 1.63
        elif self.CIB_model == 'mmDL_fit':  # fit by hand, see : /root/download/ILC_ALL/ILC_MINE/fit_cib.ipynb
            Tdust_CIB = 8.35
            beta_CIB = 1.59

        nu0_CIB_ghz = 353.0    #CIB pivot frequency [GHz]
        kT_e_keV = 5.0         #electron temperature for relativistic SZ evaluation [keV] (for reference, 5 keV is rough temperature for a 3x10^14 Msun/h cluster at z=0 (e.g., Arnaud+2005))
        nu0_radio_ghz = 150.0  #radio pivot frequency [GHz]
        beta_radio = -0.5      #radio power-law index

        nu_ghz = freqs
        nu = 1.e9*np.asarray(nu_ghz).astype(float)
        X_CIB = hplanck*nu/(kboltz*Tdust_CIB)
        nu0_CIB = nu0_CIB_ghz*1.e9
        X0_CIB = hplanck*nu0_CIB/(kboltz*Tdust_CIB)
        resp = (nu/nu0_CIB)**(3.0+(beta_CIB)) * ((np.exp(X0_CIB) - 1.0) / (np.exp(X_CIB) - 1.0)) * (ItoDeltaT(np.asarray(nu_ghz).astype(float))/ItoDeltaT(nu0_CIB_ghz))
        #resp = (nu)**(3.0+(beta_CIB)) * (1 / (np.exp(X_CIB) - 1.0)) * (ItoDeltaT(np.asarray(nu_ghz).astype(float)))
        resp[np.where(nu_ghz == None)] = 0. #this case is appropriate for HI or other maps that contain no CMB-relevant signals (and also no CIB); they're assumed to be denoted by None in nu_ghz
        return resp
    
    def get_cib_shot_noise(self):
        #from table 6 of Planck Collaboration: CIB Anisotropies with Planck
        planck_freqs = np.array([545, 353, 217, 143, 100]) #we only use 353, 217, 100
        all_freqs = np.array([545, 353, 280, 225, 217, 145, 143, 100, 93])
        shot_noise_545 = np.array([1454, 543, 135, 35, 12])
        shot_noise_353 = np.array([543, 225, 59, 15, 5.4])
        shot_noise_217 = np.array([135, 59, 16, 4.3, 1.5])
        shot_noise_143 = np.array([35, 15, 4.3, 1.2, 0.42])
        shot_noise_100 = np.array([12, 5.4, 1.5, 0.42, 0.15])
        planck_shot_noise = np.array([shot_noise_545, shot_noise_353, shot_noise_217, shot_noise_143, shot_noise_100])
        for i in range(len(planck_shot_noise)):
            for j in range(len(planck_shot_noise[0])):
                freq1, freq2 = planck_freqs[i], planck_freqs[j]
                equiv1 = u.thermodynamic_temperature(freq1 * u.GHz, Planck15.Tcmb0)
                planck_shot_noise[i][j] *= (1. * u.Jy / u.sr).to(u.uK, equivalencies=equiv1).value
                equiv2 = u.thermodynamic_temperature(freq2 * u.GHz, Planck15.Tcmb0)
                planck_shot_noise[i][j] *= (1. * u.Jy / u.sr).to(u.uK, equivalencies=equiv2).value
        
        #get planck x (planck and SO)
        planck_shot_noise_log = np.log(planck_shot_noise)
        planck_cross_all_shot_noise = np.zeros((5, len(all_freqs)))
        for i, sn in enumerate(planck_shot_noise_log):
            f = interp1d(planck_freqs, sn, fill_value='extrapolate')
            planck_cross_all_shot_noise[i] = np.exp(f(all_freqs))
        
        #get (planck and SO) x (planck and SO)
        all_cross_planck_shot_noise = planck_cross_all_shot_noise.T
        all_cross_planck_shot_noise_log = np.log(all_cross_planck_shot_noise)
        all_shot_noise = np.zeros((len(all_freqs), len(all_freqs)))
        for i, sn in enumerate(all_cross_planck_shot_noise_log):
            f = interp1d(planck_freqs, sn, fill_value='extrapolate')
            all_shot_noise[i] = np.exp(f(all_freqs))
        
        indices_to_delete = [i for i in range(len(all_shot_noise)) if all_freqs[i] not in self.freqs]
        all_shot_noise = np.delete(all_shot_noise, indices_to_delete, axis=0)
        all_shot_noise = np.delete(all_shot_noise, indices_to_delete, axis=1)
        all_shot_noise = np.flip(all_shot_noise)
        self.cib_shot_noise = all_shot_noise
        return all_shot_noise

    
    def populate_all_spectra(self, plot=False):
        
        #populate all_spectra from files, index as all_spectra[a][i][b][j] 
        
        cib_shot_noise = self.get_cib_shot_noise()
        
        for a, comp1 in enumerate(self.comps):
            for i, freq1 in enumerate(self.freqs):
                plt.clf()
                for b, comp2 in enumerate(self.comps):
                    for j, freq2 in enumerate(self.freqs):
                        if comp1==comp2=='kSZ':
                            spectrum = self.get_ksz_auto_spectra()
                        elif comp1=='kSZ' or comp2=='kSZ': #kSZ cross anything is 0
                            continue
                        elif comp1==comp2=='noise':
                            if i==j:
                                if self.noise is None:
                                    spectrum = self.get_noise_auto_spectra(self.freqs[i])
                                else:
                                    spectrum = self.noise[i]
                            else:
                                spectrum = np.zeros(self.ellmax+1)
                        elif comp1=='noise' or comp2=='noise': #noise cross anything is 0
                            continue
                        elif (comp1=='CMB' or comp2=='CMB') and not (comp1=='CMB' and comp2=='CMB'): #CMB cross anything is 0
                            continue
                        elif (comp1=='radio' or comp2=='radio') and not (comp1=='radio' and comp2=='radio'): #radio cross anything is 0
                            continue
                        else:
                            try:
                                file = open(self.get_file_name(self.base_dir, freq1, freq2, comp1, comp2), 'r')
                            except FileNotFoundError:
                                file = open(self.get_file_name(self.base_dir, freq2, freq1, comp2, comp1), 'r')
                            ells_here, spectrum = np.loadtxt(file)
                            spectrum = self.cubic_interp(ells_here, spectrum)
                            spectrum = spectrum/(self.ells*(self.ells+1))*(2*np.pi)
                            spectrum[0] = 0
                            if comp1==comp2=='CIB': #add CIB shot noise
                                spectrum += cib_shot_noise[i][j]
                        self.all_spectra[a][i][b][j] = spectrum
                        if plot:
                            plt.plot(self.ells, self.ells*(self.ells+1)*spectrum/(2*np.pi), label=f'{comp1}_{freq1}GHz, {comp2}_{freq2}GHz')
                if plot:
                    plt.xlabel(r'$\ell$')
                    plt.ylabel(r'$D_{\ell}$')
                    plt.legend()
                    plt.show()
        return self.all_spectra
    
    def populate_comp_cross_g_spectra(self, plot=False):
        
        #populate comp_cross_g_spectra, index as comp_cross_g_spectra[component][freq][0-2 for unWISE blue, green, or red]

        for a, comp in enumerate(self.comps):
            if comp=='kSZ' or comp=='noise' or comp=='CMB' or comp=='radio':
                continue
            for i, freq in enumerate(self.freqs):
                for b, color in enumerate(['g_blue', 'g_green', 'g_red']):
                    file = open(f'{self.base_dir}/ell_dl_{freq}x{freq}_GHz_{comp}xg_wLensmag_{color[2:]}.txt', 'r')
                    ells_here, spectrum = np.loadtxt(file)
                    f_spectrum = interp1d(ells_here, spectrum, fill_value="extrapolate")
                    spectrum = f_spectrum(self.ells)
                    spectrum = spectrum/(self.ells*(self.ells+1))*(2*np.pi)
                    spectrum[0] = 0
                    self.comp_cross_g_spectra[a][i][b] = spectrum
                    if plot:
                        plt.plot(self.ells, self.ells*(self.ells+1)*spectrum/(2*np.pi), label=f'{comp} {freq} GHz x {color}')
        if plot:
            plt.clf()
            plt.xlabel(r'$\ell$')
            plt.ylabel(r'$D_{\ell}$')
            plt.legend()
            plt.show()
        return self.comp_cross_g_spectra 
    
    def populate_gg_spectrum(self, plot=False):

        #populate gg auto- and cross-spectra index as gg_spectrum[a][b][l] where a and b index unWISE blue, green, red

        plt.clf()
        for a, color1 in enumerate(['blue', 'green', 'red']):
            for b, color2 in enumerate(['blue', 'green', 'red']):
                file = open(f'{self.base_dir}/ell_dl_gxg_wLensmag_{color1}_x_{color2}.txt', 'r')
                ells_gg, gg = np.loadtxt(file)
                f_gg = interp1d(ells_gg, gg, fill_value="extrapolate")
                gg = f_gg(self.ells)
                if color1==color2=='blue':
                    self.blue_gg = gg
                gg_shot_noise = self.gg_shot_noise[a][b]
                if plot:
                    plt.plot(self.ells, gg, label='gg signal')
                    plt.plot(self.ells, self.ells*(self.ells+1)*gg_shot_noise/(2*np.pi), label='gg shot noise')
                    plt.plot(self.ells, gg+self.ells*(self.ells+1)*gg_shot_noise/(2*np.pi), label='total gg spectrum')
                    plt.title(f'gg {color1} x {color2} Spectrum')
                    plt.xlabel(r'$\ell$')
                    plt.ylabel(r'$D_{\ell}$ [$\mu K^2$]')
                    plt.yscale('log')
                    plt.legend()
                    plt.show()
                self.gg_spectrum[a][b] = gg/(self.ells*(self.ells+1))*(2*np.pi)
                self.gg_spectrum[a][b][0] = 0
                self.gg_spectrum[a][b] += gg_shot_noise
        return self.gg_spectrum
    
    def get_cls(self, plot=False):
        self.populate_all_spectra(plot=plot)
        self.populate_comp_cross_g_spectra(plot=plot)
        self.populate_gg_spectrum(plot=plot)
        self.Clij = np.einsum('aibjl->ijl', self.all_spectra) #index as Clij[i][j][l]
        self.Clig = np.einsum('aibl->ibl', self.comp_cross_g_spectra) #index as Clig[i][b][l]
        self.Clgg = self.gg_spectrum #index as Clgg[a][b][l]
        if plot:
            plt.clf()
            for i in range(len(freqs)):
                for j in range(len(freqs)):
                    plt.plot(self.ells, self.ells*(self.ells+1)*self.Clij[i][j]/(2*np.pi), label=r'$D_{\ell}$ '+f'{freqs[i]}, {freqs[j]}')
            plt.xlabel(r'$\ell$')
            plt.ylabel(r'$D_{\ell}$')
            plt.title(r'$D_{\ell}^{ij}$')
            plt.yscale('log')
            plt.legend()
            plt.show()

            plt.clf()
            for i in range(len(freqs)):
                for b in range(3):
                    plt.plot(self.ells, self.ells*(self.ells+1)*self.Clig[i][b]/(2*np.pi), label=f'{freqs[i]} GHz, color {b}')
            plt.xlabel(r'$\ell$')
            plt.ylabel(r'$D_{\ell}$')
            plt.title(r'$D_{\ell}^{ig}$')
            plt.legend()
            plt.show()

            plt.clf()
            for a in range(3):
                for b in range(3):
                    plt.plot(self.ells, self.ells*(self.ells+1)*self.Clgg[a][b]/(2*np.pi), label=f'colors {a} and {b}')
            plt.xlabel(r'$\ell$')
            plt.ylabel(r'$D_{\ell}$')
            plt.legend()
            plt.title(r'$D_{\ell}^{gg}$')
            plt.yscale('log')
            plt.show()
        return self.Clij, self.Clig, self.Clgg