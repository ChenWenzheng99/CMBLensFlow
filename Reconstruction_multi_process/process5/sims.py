import os
import sys
import numpy as np
import healpy as hp
import lenspyx
import multiprocessing as mp
import argparse

sys.path.insert(0, './')
from one import *
from utils import uKamin2uKpix, bl_eft, bl, apodize_mask

'''
该脚本包含以下三个函数：
1、lensed_cmbs: 输入Cl,nside,FWHM,调用healpy和lenspyx产生lensed TQU map 和 P map
2、noise: 输入nlev,seed,利用高斯分布模拟产生noise map
3、ninv: 输入nlev,得到Noise inveres pixel varance for inhomogeneous filtering.
'''

parser = argparse.ArgumentParser(description='Simulation for AliCPT Lensing')   #创建一个解析器对象，用于定义和解析命令行参数。
parser.add_argument('-np', dest='np', type=int, default=1, help='Number of processes')   #添加要解析的命令行参数的规则  # -np 传递进程数
args = parser.parse_args()   #解析命令行参数，并返回一个包含解析结果的对象 args

def lensed_cmbs(cls, nside, savePath, fwhm_f=[], nrms_f=None, lmax=4096, dlmax=1024, facres=-1, seed=0):
    """ Lensed CMB TQU maps random realiazation.

         * cls      : dict which contiains input power spectra
         * nside    : nside
         * savePath : directory where you store the simulations
         * lmax     : max ell
         * dlmax    : dmax ell
         * fwhm     : Full Width Half Maximum (Beam size)
         * seed     : array which contains random seed
    """
    assert 'OMP_NUM_THREADS' in os.environ.keys(), 'Check your env variable OMP_NUM_THREADS' #OMP_NUM_THREADS用于控制并行计算库（例如OpenMP）的线程数量，在.bashrc中加入export OMP_NUM_THREADS=4

    fname_TQU = 'map_TQU_%d_%04d.fits'
    fname_P = 'map_P_%d_%04d.fits'
    
    # Lensed maps realization. Correlations between PTE not considered.
    # transfer function NOTE pixwin func is included as well
    if len(fwhm_f) == 0 or 1:
        fwhm_f = fwhm_f[0] if fwhm_f else 0.
        transf = bl(fwhm_f, nside=nside, lmax=lmax, pixwin=True)
    else:
        assert nrms_f
        transf = bl_eft(nrms_f, fwhm_f, nside=nside, lmax=lmax, pixwin=True)  #Effective beam function

    np.random.seed(seed)  
    #seed 用于初始化 NumPy 随机数生成器，当设置了相同的 seed 值时，np.random 将使用相同的初始状态来生成随机数序列，这生成相同的随机数。通过更改 seed 值，可以创建不同的随机数序列，例见1.ipynb

    # Unlensed TQUP alms
    plm = hp.synalm(cls['pp'], lmax=lmax + dlmax, new=True, verbose=False)
    Pmap = hp.alm2map(plm, nside, verbose=False)
    dlm = hp.almxfl(plm, np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2)))
    tlm_unl, elm_unl, blm_unl = hp.synalm(  [cls['tt'], cls['ee'], cls['bb'], cls['te']], lmax=lmax + dlmax, new=True, verbose=False)
    # NOTE we only consider lensing induced E->B modes
    
    # Lensed TQU maps          
    geom_info = ('healpix', {'nside':nside}) 
    Tlen = lenspyx.lensing.alm2lenmap(tlm_unl, dlm,  geometry=geom_info, verbose=False)#alm2lenmap：alm_unl * dlm -> lensed map
    Qlen, Ulen = lenspyx.lensing.alm2lenmap_spin(elm_unl, dlm, 2, geometry=geom_info, verbose=False)
    '''
    this is the original code, but it is not suitable for the new version of lenspyx, it is suitable for lenspyx 2.0.0
    Tlen = lenspyx.alm2lenmap(tlm_unl, [dlm, None], nside, facres=facres, verbose=False)
    Qlen, Ulen = lenspyx.alm2lenmap_spin([elm_unl, None], [dlm, None], nside, 2, geometry=geom_info, verbose=False)   #这里QU的spin=2,要用alm2lenmap_spin
    '''
    tlm_len = hp.map2alm(Tlen, lmax=lmax)
    elm_len, blm_len = hp.map2alm_spin([Qlen, Ulen], 2, lmax=lmax)
    
    # Convolution with transfer function    #tranf = H_l*B_l, 见PLANCK2015 (A.4)，亦见PLANCK2013 (38)式
    Tlen = hp.alm2map(hp.almxfl(tlm_len, transf, inplace=True), nside, verbose=False)
    Qlen, Ulen = hp.alm2map_spin([hp.almxfl(elm_len, transf), hp.almxfl(blm_len, transf)], nside, 2, lmax)

    # Save fits File
    hp.write_map(os.path.join(savePath, fname_TQU % (nside, seed)), [Tlen, Qlen, Ulen], overwrite=True)
    hp.write_map(os.path.join(savePath, fname_P % (nside, seed)), Pmap, overwrite=True)
    print(1)

def noise(nlev, nside, savePath, seed=0, fwhm_f=None):   #The inhomogeneous instrumental noises is generated based on AliCPT noise variance map.
    """ Noise simulations, we include
            - white noise
            - noise realization according to given noise variance map
            - noise realization and combination of multi-channels

        * nlev      : it depends ...
        * nside     : nside
        * savePath  : directory to save data
        * seed      : random seed
        * fwhm_f    : only for multi-channel combination

        All AliCPT's detectors are polarized and thus for simplicity
        nlev_Q = nlev_T * sqrt(2)
    """
    assert isinstance(nlev, list or float or int)   #检查输入的nlev数据类型
    npix = hp.nside2npix(nside)
    fname = 'map_noise_nside%d_%04d.fits'

    np.random.seed(seed + np.random.randint(99999999, size=1))
    m = np.random.normal(size=(3, npix)) * hp.read_map(nlev[0], verbose=False) \
                * np.array([1, 2 ** 0.5, 2 ** 0.5]).reshape(3,1)     #三行分别是T,Q,U的
    #variance map 里存的不是variance,而是standard deviation(标准差)，这里生成了符合noise标准差的高斯分布的随机数，得到variance map

    hp.write_map(os.path.join(savePath, fname % (nside, seed)), m, overwrite=True)   #返回noise map

def noises(nlev, nside, savePath, seed=0, fwhm_f=None):
    """ Noise simulations, we include
            - white noise
            - noise realization according to given noise variance map
            - noise realization and combination of multi-channels

        * nlev      : it depends ...
        * nside     : nside
        * savePath  : directory to save data
        * seed      : random seed
        * fwhm_f    : only for multi-channel combination

        All AliCPT's detectors are polarized and thus for simplicity
        nlev_Q = nlev_T * sqrt(2)
    """
    npix = hp.nside2npix(nside)
    fname = 'map_noise_nside%d_%04d.fits'

    m = np.random.normal(size=(3, npix)) * uKamin2uKpix(nlev, npix) \
    * np.array([1, 2 ** 0.5, 2 ** 0.5]).reshape(3,1)

    hp.write_map(os.path.join(savePath, fname % (nside, seed)), m, overwrite=True)


def ninv(nlev, savePath, fwhm_f=None):
    """ Noise inveres pixel variance for inhomogeneous filtering.

        * nlev      : it dependes ...

        All AliCPT's detectors are polarized and thus for simplicity
        nlev_Q = nlev_T * sqrt(2)
    """
    assert isinstance(nlev, list)
    nrms_f = [hp.read_map(path, verbose=False) for path in nlev]
    fname_ninv_t = 'ninv_t.fits'
    fname_ninv_p = 'ninv_p.fits'


    # TODO Gaussian smoothing the variance map
    if len(nlev) == 1:    #如果nlev列表只包含一个噪声水平，即只有nlev_t,直接计算。这里的1不是指列表只有一个数，而是只有nlev_t的一组列表元素，shape为(1, 12582912)
        ninv_t = nrms_f[0] ** 2  #nrms_f列表为[[0. 0. 0. ... 0. 0. 0.]]，因此nrms_f[0]：[0. 0. 0. ... 0. 0. 0.]表示nlev_t.
        ninv_t[ninv_t!=0] = ninv_t[ninv_t!=0] ** -1  #逆方差
        ninv_p = ninv_t / 2.
        #若输入的nlevmap有t,p，那么nrms_f列表为[[0. 0. 0. ... 0. 0. 0.]，[0. 0. 0. ... 0. 0. 0.]]，nrms_f[0]和nrms_f[1]分别为nlev_t和nlev_p
    else:
        #这部分原先用于两channel的情况，现在不用了
        # assert fwhm_f and fwhm_c
        # nside = hp.npix2nside(len(nrms_f[0]))
        # mask_b = np.where(nrms_f[0] != 0, 1, 0)
        # w_f = np.array(wl_f(nrms_f, fwhm_f, fwhm_c, pixwin=True))
        # ninv_t = vmaps2vmap_I([ nrms ** 2 for nrms in nrms_f ], w_f, nside) * mask_b
        # ninv_t[ninv_t!=0] = ninv_t[ninv_t!=0] ** -1
        # ninv_p = ninv_t / 2.

        assert fwhm_f   #如果nlev列表包含nlev_t和nlev_p，
        nlev_c, transf = bl_eft(nrms_f, fwhm_f, lmax=lmax, pixwin=True, ret_nlev=True)
        npix = len(nrms_f[0])
        mask = apodize_mask(nrms_f[0])
        ninv_t = uKamin2uKpix(nlev_c, npix) ** -2 * mask  ############ 乘mask物理含义，见 PLANCK2015 (A.7) #################
        ninv_p = ninv_t / 2.
        

    hp.write_map(os.path.join( savePath, fname_ninv_t ), ninv_t, overwrite=True)
    hp.write_map(os.path.join( savePath, fname_ninv_p ), ninv_p, overwrite=True)
    
#noises(10., 1024, "/disk1/home/hanjk/Noise_ALI_IHEP_20200730_48/sims/", seed=0, fwhm_f=None)

if __name__ == '__main__':  #如果脚本被直接执行（而不是被导入为模块），则进入if __name__ == '__main__':代码块。
    # MultiProcessing
    pool = mp.Pool(processes=args.np)  #创建一个进程池（mp.Pool），并指定最大进程数为args.np，即通过命令行参数传递的进程数。
    #for seed in seeds:      #使用for循环遍历名为seeds的迭代器中的每个元素，其中seeds可能是一个包含随机种子的列表或迭代器。
        #pool.apply_async(lensed_cmbs, args=(cls_in, nside, savePath_cmbs),   #对于每个随机种子seed，使用pool.apply_async方法异步地执行两个函数：lensed_cmbs 和 noise
                #kwds={'lmax':lmax, 'dlmax':dlmax, 'fwhm_f':fwhm_f, 'nrms_f':nlev, 'seed':seed})
        #pool.apply_async(noise, args=(nlev, nside, savePath_noise),
                #kwds={'fwhm_f':fwhm_f, 'seed':seed})

    pool.close()  #执行完所有的异步任务后，使用pool.close()关闭进程池
    pool.join()   #使用pool.join()等待所有进程完成任务

    ninv(nlev, savePath_ninv, fwhm_f=fwhm_f)  #最后，调用ninv函数，计算噪声逆像素方差(结果已在运行函数时保存)
