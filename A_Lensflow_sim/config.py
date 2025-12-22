DEFAULT_CONFIG = {
    'nside': 1024,
    'lmax' : 3072,
    'freqs': [27, 39, 93 ,145, 225, 280],  # GHz
    'beam' : [7.4, 5.1, 2.2, 1.4, 1.0, 0.9],
    'mask_dir': None,

    ######################### CMB #########################
    'cmb': True,                    # Switch for CMB simulation
    'cmb_params': {
        'fid_source':   'FFP10',    # Default FFP10, can also be 'CAMB' or 'custom', see 'cmb_making.py' for details
        'fid_type':     'cl',       # Input fiducial type, 'cl' or 'map'
        'dlmax':        None,       # Maximum ell for CMB lensing potential
        'seed':         42,         # Random seed for CMB simulation
        'epsilon':      None,
        'if_lensing':   True,
        'phi_dir':      None,
        'unlensed_dir':  None,
        'CAMB_params':  None,
        'lenspyx_geom': 'healpix',
    },

    ######################### Foreground #########################
    'foreground': True,        # Switch for Foreground simulation   
    'foreground_params': {
    'comps': ["d9", "s4", "a1", "f1", "cib1", "tsz1", "ksz1", "d1","s1", "f1", "cib1", "tsz1"],
    'unit': "uK_CMB",
    'cordinate': 'G',
    'if_gaussian': True,       # Whether use Gaussian foregrounds, notice this only includes dust and synchrotron.
    'seed': [100, 101, 102, 103, 104, 105]
    },


    ######################### Noise #########################
    'noise': True,                  # Switch for Noise simulation
    'noise_params': {
    'nlev': [52, 27, 5.8, 6.3, 15, 37],  # uK-arcmin
    'noise_model': 'homogeneous',   # white noise from pixel level
    'seed': [24, 25, 26, 27, 28, 29],
    'N_red_T': None,                # SO-LAT : [100, 39, 230, 1500, 17000, 31000]
    'l_knee_T': None,               # SO-LAT : [1000, 1000, 1000, 1000, 1000, 1000]
    'alpha_knee_T': None,           # SO-LAT : [-3.5, -3.5, -3.5, -3.5, -3.5, -3.5]
    'n_red_P': None,                # SO-LAT : [52, 27, 5.8, 6.3, 15, 37]
    'l_knee_P': None,               # SO-LAT : [700, 700, 700, 700, 700, 700]
    'alpha_knee_P': None,           # SO-LAT : [-1.4, -1.4, -1.4, -1.4, -1.4, -1.4]
    'if_red': False,
    'scale_factor': 2,
    },


}
