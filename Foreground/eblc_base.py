import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

class EBLeakageCorrection:
    def __init__(self, m, lmax, nside, mask, post_mask, method:str='cutqufitqu', check_res:bool=False, n_iter=3):
        ''' m are IQU maps of one frequency '''
        self.m = m
        self.lmax = lmax
        self.nside = nside
        self.mask = mask
        self.post_mask = post_mask
        self.method = method
        self.check_res = check_res
        self.n_iter = n_iter

    def add_mask(self, m):
        return m * self.mask

    def add_post_mask(self, m):
        return m * self.post_mask

    def check_eblc_result(self, crt_b, tmp_b, cln_b, rot:list=[100,50,0], vmin=-0.6, vmax=0.6, cmap='jet'):
        hp.orthview(hp.ma(crt_b, badval=0), rot=[100,50,0], half_sky=True,sub=(1,3,1), title='corrupted', min=vmin, max=vmax, cmap=cmap, badcolor='white')
        hp.orthview(hp.ma(tmp_b, badval=0), rot=[100,50,0],  half_sky=True,sub=(1,3,2), title='template', min=vmin, max=vmax, cmap=cmap, badcolor='white')
        hp.orthview(hp.ma(cln_b, badval=0), rot=[100,50,0],  half_sky=True,sub=(1,3,3), title='cleaned', min=vmin, max=vmax, cmap=cmap, badcolor='white')
        plt.show()

    def zzr(self, lmax, nside):
        def calc_crt_tmp_b(lmax, nside):
            crt_alms = hp.map2alm(self.add_mask(self.m), lmax=lmax)
            crt_alm_t, crt_alm_e, crt_alm_b = [crt_alm for crt_alm in crt_alms]
            self.crt_b = self.add_mask(hp.alm2map(crt_alm_b, nside=nside))

            self.tmp_fml_e = self.add_mask(hp.alm2map([crt_alm_t, crt_alm_e, np.zeros_like(crt_alm_t)], nside=nside))
            tmp_alms = hp.map2alm(self.tmp_fml_e, lmax=lmax)
            tmp_alm_t, tmp_alm_e, tmp_alm_b = [tmp_alm for tmp_alm in tmp_alms]
            self.tmp_b = self.add_mask(hp.alm2map(tmp_alm_b, nside=nside))

        def linear_fitting():
            coeffs = np.polyfit(self.tmp_b, self.crt_b, 1)
            slope, intercept = coeffs
            print(f'{slope=}, {intercept=}')
            self.cln_b = self.crt_b - slope * self.tmp_b

        def apo_for_next_step():
            self.crt_b = self.add_post_mask(self.crt_b)
            self.tmp_b = self.add_post_mask(self.tmp_b)
            self.cln_b = self.add_post_mask(self.cln_b)
        calc_crt_tmp_b(lmax, nside)
        linear_fitting()
        self.check_res and self.check_eblc_result(self.crt_b, self.tmp_b, self.cln_b)
        apo_for_next_step()

    def fullqufitb(self, lmax, nside):
        def calc_crt_tmp_b(lmax, nside):
            crt_alms = hp.map2alm(self.add_mask(self.m), lmax=lmax)
            crt_alm_t, crt_alm_e, crt_alm_b = [crt_alm for crt_alm in crt_alms]

            self.crt_full_qu = hp.alm2map([crt_alm_t, np.zeros_like(crt_alm_t), crt_alm_b], nside=nside)
            self.crt_b = self.add_mask(hp.alm2map(hp.map2alm(self.crt_full_qu, lmax=lmax)[2], nside=nside))

            self.tmp_fml_e = self.add_mask(hp.alm2map([crt_alm_t, crt_alm_e, np.zeros_like(crt_alm_t)], nside=nside))
            tmp_alms = hp.map2alm(self.tmp_fml_e, lmax=lmax)
            tmp_alm_t, tmp_alm_e, tmp_alm_b = [tmp_alm for tmp_alm in tmp_alms]

            self.tmp_full_qu = hp.alm2map([tmp_alm_t, np.zeros_like(tmp_alm_t), tmp_alm_b], nside=nside)
            self.tmp_b = self.add_mask(hp.alm2map(hp.map2alm(self.tmp_full_qu, lmax=lmax)[2], nside=nside))

        def linear_fitting():
            coeffs = np.polyfit(self.tmp_b, self.crt_b, 1)
            slope, intercept = coeffs
            print(f'{slope=}, {intercept=}')
            self.cln_b = self.crt_b - slope * self.tmp_b

        def apo_for_next_step():
            self.crt_b = self.add_post_mask(self.crt_b)
            self.tmp_b = self.add_post_mask(self.tmp_b)
            self.cln_b = self.add_post_mask(self.cln_b)
        calc_crt_tmp_b(lmax, nside)
        linear_fitting()
        self.check_res and self.check_eblc_result(self.crt_b, self.tmp_b, self.cln_b)
        apo_for_next_step()

    def fullqufitqu(self, lmax, nside):
        def calc_crt_tmp_qu(lmax, nside):
            crt_alms = hp.map2alm(self.add_mask(self.m), lmax=lmax)
            crt_alm_t, crt_alm_e, crt_alm_b = [crt_alm for crt_alm in crt_alms]
            self.crt_full_qu = hp.alm2map([crt_alm_t, np.zeros_like(crt_alm_t), crt_alm_b], nside=nside)

            self.tmp_fml_e = self.add_mask(hp.alm2map([crt_alm_t, crt_alm_e, np.zeros_like(crt_alm_t)], nside=nside))
            tmp_alms = hp.map2alm(self.tmp_fml_e, lmax=lmax)
            tmp_alm_t, tmp_alm_e, tmp_alm_b = [tmp_alm for tmp_alm in tmp_alms]
            self.tmp_full_qu = hp.alm2map([tmp_alm_t, np.zeros_like(tmp_alm_t), tmp_alm_b], nside=nside)

        def linear_fitting():
            coeffs = np.polyfit(self.tmp_full_qu[1:2].flatten(), self.crt_full_qu[1:2].flatten(), 1)
            slope, intercept = coeffs
            print(f'{slope=}, {intercept=}')
            self.cln_full_qu = self.crt_full_qu - slope * self.tmp_full_qu

        def calc_crt_tmp_cln_b():
            self.crt_b = self.add_mask(hp.alm2map(hp.map2alm(self.crt_full_qu, lmax=lmax)[2], nside=nside))
            self.tmp_b = self.add_mask(hp.alm2map(hp.map2alm(self.tmp_full_qu, lmax=lmax)[2], nside=nside))
            self.cln_b = self.add_mask(hp.alm2map(hp.map2alm(self.cln_full_qu, lmax=lmax)[2], nside=nside))

        def apo_for_next_step():
            self.crt_b = self.add_post_mask(self.crt_b)
            self.tmp_b = self.add_post_mask(self.tmp_b)
            self.cln_b = self.add_post_mask(self.cln_b)
        calc_crt_tmp_qu(lmax, nside)
        linear_fitting()
        calc_crt_tmp_cln_b()
        self.check_res and self.check_eblc_result(self.crt_b, self.tmp_b, self.cln_b)
        apo_for_next_step()

    def cutqufitb(self, lmax, nside):
        def calc_crt_tmp_b(lmax, nside):
            crt_alms = hp.map2alm(self.add_mask(self.m), lmax=lmax)
            crt_alm_t, crt_alm_e, crt_alm_b = [crt_alm for crt_alm in crt_alms]

            self.crt_cut_qu = self.add_mask(hp.alm2map([crt_alm_t, np.zeros_like(crt_alm_t), crt_alm_b], nside=nside))
            self.crt_b = self.add_mask(hp.alm2map(hp.map2alm(self.crt_cut_qu, lmax=lmax)[2], nside=nside))

            self.tmp_fml_e = self.add_mask(hp.alm2map([crt_alm_t, crt_alm_e, np.zeros_like(crt_alm_t)], nside=nside))
            tmp_alms = hp.map2alm(self.tmp_fml_e, lmax=lmax)
            tmp_alm_t, tmp_alm_e, tmp_alm_b = [tmp_alm for tmp_alm in tmp_alms]

            self.tmp_cut_qu = self.add_mask(hp.alm2map([tmp_alm_t, np.zeros_like(tmp_alm_t), tmp_alm_b], nside=nside))
            self.tmp_b = self.add_mask(hp.alm2map(hp.map2alm(self.tmp_cut_qu, lmax=lmax)[2], nside=nside))

        def linear_fitting():
            coeffs = np.polyfit(self.tmp_b, self.crt_b, 1)
            slope, intercept = coeffs
            print(f'{slope=}, {intercept=}')
            self.cln_b = self.crt_b - slope * self.tmp_b

        def apo_for_next_step():
            self.crt_b = self.add_post_mask(self.crt_b)
            self.tmp_b = self.add_post_mask(self.tmp_b)
            self.cln_b = self.add_post_mask(self.cln_b)
        calc_crt_tmp_b(lmax, nside)
        linear_fitting()
        self.check_res and self.check_eblc_result(self.crt_b, self.tmp_b, self.cln_b)
        apo_for_next_step()

    def cutqufitqu(self, m, lmax, nside):
        def calc_crt_tmp_qu(lmax, nside):
            crt_alms = hp.map2alm(self.add_mask(m), lmax=lmax)
            crt_alm_t, crt_alm_e, crt_alm_b = [crt_alm for crt_alm in crt_alms]
            self.crt_cut_qu = self.add_mask(hp.alm2map([crt_alm_t, np.zeros_like(crt_alm_t), crt_alm_b], nside=nside))

            self.tmp_fml_e = self.add_mask(hp.alm2map([crt_alm_t, crt_alm_e, np.zeros_like(crt_alm_t)], nside=nside))
            tmp_alms = hp.map2alm(self.tmp_fml_e, lmax=lmax)
            tmp_alm_t, tmp_alm_e, tmp_alm_b = [tmp_alm for tmp_alm in tmp_alms]
            self.tmp_cut_qu = self.add_mask(hp.alm2map([tmp_alm_t, np.zeros_like(tmp_alm_t), tmp_alm_b], nside=nside))

        def linear_fitting():
            coeffs = np.polyfit(self.tmp_cut_qu[1:2].flatten(), self.crt_cut_qu[1:2].flatten(), 1)
            slope, intercept = coeffs
            print(f'{slope=}, {intercept=}')
            self.cln_cut_qu = self.crt_cut_qu - slope * self.tmp_cut_qu

        def calc_crt_tmp_cln_b():
            self.crt_b = self.add_mask(hp.alm2map(hp.map2alm(self.crt_cut_qu, lmax=lmax)[2], nside=nside))
            self.tmp_b = self.add_mask(hp.alm2map(hp.map2alm(self.tmp_cut_qu, lmax=lmax)[2], nside=nside))
            self.cln_b = self.add_mask(hp.alm2map(hp.map2alm(self.cln_cut_qu, lmax=lmax)[2], nside=nside))

        def apo_for_next_step():
            self.crt_b = self.add_post_mask(self.crt_b)
            self.tmp_b = self.add_post_mask(self.tmp_b)
            self.cln_b = self.add_post_mask(self.cln_b)

        calc_crt_tmp_qu(lmax, nside)
        linear_fitting()
        calc_crt_tmp_cln_b()
        self.check_res and self.check_eblc_result(self.crt_b, self.tmp_b, self.cln_b)
        apo_for_next_step()

    def iterative_eblc(self, m, lmax, nside, n_iter=5):
        def calc_crt_qu(lmax, nside):
            crt_alms = hp.map2alm(self.add_mask(m), lmax=lmax)
            crt_alm_t, crt_alm_e, crt_alm_b = [crt_alm for crt_alm in crt_alms]
            self.crt_cut_qu = self.add_mask(hp.alm2map([crt_alm_t, np.zeros_like(crt_alm_t), crt_alm_b], nside=nside))
        def calc_next_cut_qu(cut_qu):
            iter_alms = hp.map2alm(self.add_mask(cut_qu), lmax=lmax)
            iter_alm_t, iter_alm_e, iter_alm_b = [iter_alm for iter_alm in iter_alms]
            self.iter_cut_qu = self.add_mask(hp.alm2map([iter_alm_t, np.zeros_like(iter_alm_t), iter_alm_b], nside=nside))
        def iter_qu_fml(n_iter):
            calc_next_cut_qu(self.crt_cut_qu)
            for i in range(n_iter):
                calc_next_cut_qu(self.iter_cut_qu)

        def calc_crt_cln_b():
            self.crt_b = self.add_mask(hp.alm2map(hp.map2alm(self.crt_cut_qu, lmax=lmax)[2], nside=nside))
            self.cln_b = self.add_mask(hp.alm2map(hp.map2alm(self.iter_cut_qu, lmax=lmax)[2], nside=nside))
            self.tmp_b = None

        def apo_for_next_step():
            self.crt_b = self.add_post_mask(self.crt_b)
            self.cln_b = self.add_post_mask(self.cln_b)

        calc_crt_qu(lmax, nside)
        iter_qu_fml(n_iter)
        calc_crt_cln_b()
        apo_for_next_step()

    def fit_iter_crt_tmp(self, lmax, nside, n_iter_crt=9, n_iter_tmp=9):
        ''' might be wrong !!! '''
        def calc_crt_tmp_qu(lmax, nside):
            crt_alms = hp.map2alm(self.add_mask(self.m), lmax=lmax)
            crt_alm_t, crt_alm_e, crt_alm_b = [crt_alm for crt_alm in crt_alms]
            self.crt_cut_qu = self.add_mask(hp.alm2map([crt_alm_t, np.zeros_like(crt_alm_t), crt_alm_b], nside=nside))

            self.tmp_fml_e = self.add_mask(hp.alm2map([crt_alm_t, crt_alm_e, np.zeros_like(crt_alm_t)], nside=nside))
            tmp_alms = hp.map2alm(self.tmp_fml_e, lmax=lmax)
            tmp_alm_t, tmp_alm_e, tmp_alm_b = [tmp_alm for tmp_alm in tmp_alms]
            self.tmp_cut_qu = self.add_mask(hp.alm2map([tmp_alm_t, np.zeros_like(tmp_alm_t), tmp_alm_b], nside=nside))

        def calc_next_cut_qu(cut_qu):
            iter_alms = hp.map2alm(self.add_mask(cut_qu), lmax=lmax)
            iter_alm_t, iter_alm_e, iter_alm_b = [iter_alm for iter_alm in iter_alms]
            iter_cut_qu = self.add_mask(hp.alm2map([iter_alm_t, np.zeros_like(iter_alm_t), iter_alm_b], nside=nside))
            return iter_cut_qu

        def iter_crt_qu_fml(n_iter_crt):
            self.iter_crt_qu = calc_next_cut_qu(self.crt_cut_qu)
            for i in range(n_iter_crt):
                self.iter_crt_qu = calc_next_cut_qu(self.iter_crt_qu)

        def iter_tmp_qu_fml(n_iter_tmp):

            self.iter_tmp_qu = calc_next_cut_qu(self.tmp_cut_qu)
            for i in range(n_iter_tmp):
                self.iter_tmp_qu = calc_next_cut_qu(self.iter_tmp_qu)

        def linear_fitting():
            coeffs = np.polyfit(self.iter_tmp_qu[1:2].flatten(), self.iter_crt_qu[1:2].flatten(), 1)
            slope, intercept = coeffs
            print(f'{slope=}, {intercept=}')
            self.cln_cut_qu = self.iter_crt_qu - slope * self.iter_tmp_qu

        def calc_crt_tmp_cln_b():
            self.crt_b = self.add_mask(hp.alm2map(hp.map2alm(self.crt_cut_qu, lmax=lmax)[2], nside=nside))
            self.cln_b = self.add_mask(hp.alm2map(hp.map2alm(self.cln_cut_qu, lmax=lmax)[2], nside=nside))
            self.tmp_b = None

        def apo_for_next_step():
            self.crt_b = self.add_post_mask(self.crt_b)
            self.cln_b = self.add_post_mask(self.cln_b)

        calc_crt_tmp_qu(lmax, nside)
        iter_crt_qu_fml(n_iter_crt)
        iter_tmp_qu_fml(n_iter_tmp)
        linear_fitting()
        calc_crt_tmp_cln_b()
        apo_for_next_step()

    def run_eblc(self):
        if self.method == 'zzr':
            self.zzr(lmax=self.lmax, nside=self.nside)
        if self.method == 'fullqufitb':
            self.fullqufitb(lmax=self.lmax, nside=self.nside)
        if self.method == 'fullqufitqu':
            self.fullqufitqu(lmax=self.lmax, nside=self.nside)
        if self.method == 'cutqufitb':
            self.cutqufitb(lmax=self.lmax, nside=self.nside)
        if self.method == 'cutqufitqu':
            self.cutqufitqu(m=self.m, lmax=self.lmax, nside=self.nside)
        if self.method == 'itercrtqu':
            self.iterative_eblc(self.m, lmax=self.lmax, nside=self.nside, n_iter=self.n_iter)
        if self.method == 'iterclncutqufitqu':
            self.cutqufitqu(m=self.m, lmax=self.lmax, nside=self.nside)
            self.iterative_eblc(self.cln_cut_qu, lmax=self.lmax, nside=self.nside, n_iter=self.n_iter)
        if self.method == 'fititerclnqu':
            self.fit_iter_crt_tmp(lmax=self.lmax, nside=self.nside)

        return self.crt_b, self.tmp_b, self.cln_b



