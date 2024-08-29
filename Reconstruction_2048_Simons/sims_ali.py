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
from generate_map import *

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
    fname_TQU = 'map_TQU_%d_%04d.fits'
    fname_P = 'map_P_%d_%04d.fits'

    ismap_exist = True
    try:
        pmap_exist = hp.read_map(os.path.join(savePath, fname_P % (nside, seed)))
        tmap_exist,qmap_exist,umap_exist = hp.read_map(os.path.join(savePath, fname_TQU % (nside, seed)),field=(0,1,2))
    except Exception as e:
        print(f"Simulation {seed} not found, start to simulate !")
        ismap_exist = False

    def simulate_and_save_maps():
        # Lensed maps realization. Correlations between PTE not considered.
        # transfer function NOTE pixwin func is included as well
        print(f'Noting : Start to simulate {seed}!')
        Tlen, Qlen, Ulen, Pmap = making_maps(nside, lmax=lmax, dlmax=dlmax, fwhm_f=fwhm_f,phi_map=None,pixwin=True,seed=seed)

        if np.any(Pmap > 1) or np.any(Tlen > 1000) or np.any(Qlen > 100) or np.any(Ulen > 100) :
            print(f'Warning : simulation {seed} has corrupted, need to simulate in the next round !')

        # Save fits File
        hp.write_map(os.path.join(savePath, fname_TQU % (nside, seed)), [Tlen, Qlen, Ulen], overwrite=True)
        hp.write_map(os.path.join(savePath, fname_P % (nside, seed)), Pmap, overwrite=True)
        print("Maps generated and saved.")


    if not ismap_exist:
        simulate_and_save_maps()
        return    #结束函数
    elif np.any(pmap_exist > 1) or np.any(tmap_exist > 1000) or np.any(qmap_exist > 100) or np.any(umap_exist > 100): 
        simulate_and_save_maps()
    else:
        print(f"pmap_exist {seed} is normal. Function terminated.")

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

    np.random.seed(seed)
    m = np.random.normal(size=(3, npix)) * hp.read_map(nlev[0], verbose=False) \
                * np.array([1, 2 ** 0.5, 2 ** 0.5]).reshape(3,1)     #三行分别是T,Q,U的
    #variance map 里存的不是variance,而是standard deviation(标准差)，这里生成了符合noise标准差的高斯分布的随机数，得到variance map

    hp.write_map(os.path.join(savePath, fname % (nside, seed)), m, overwrite=True)   #返回noise map

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

#lensed_cmbs(1, nside, savePath_cmbs, fwhm_f=[], nrms_f=None, lmax=4096, dlmax=1024, facres=-1, seed=seeds[58])


if __name__ == '__main__':  #如果脚本被直接执行（而不是被导入为模块），则进入if __name__ == '__main__':代码块。
    # MultiProcessing
    pool = mp.Pool(processes=args.np)  #创建一个进程池（mp.Pool），并指定最大进程数为args.np，即通过命令行参数传递的进程数。
    for seed in seeds:      #使用for循环遍历名为seeds的迭代器中的每个元素，其中seeds可能是一个包含随机种子的列表或迭代器。
        pool.apply_async(lensed_cmbs, args=(cls_in, nside, savePath_cmbs),   #对于每个随机种子seed，使用pool.apply_async方法异步地执行两个函数：lensed_cmbs 和 noise
                kwds={'lmax':lmax, 'dlmax':dlmax, 'fwhm_f':fwhm_f, 'seed':seed})
        #pool.apply_async(noise, args=(nlev, nside, savePath_noise),
                #kwds={'fwhm_f':fwhm_f, 'seed':seed})

    pool.close()  #执行完所有的异步任务后，使用pool.close()关闭进程池
    pool.join()   #使用pool.join()等待所有进程完成任务

    #ninv(nlev, savePath_ninv, fwhm_f=fwhm_f)  #最后，调用ninv函数，计算噪声逆像素方差(结果已在运行函数时保存)
