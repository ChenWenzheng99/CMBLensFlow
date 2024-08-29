import os
import sys
import numpy as np
import healpy as hp
import lenspyx
import multiprocessing as mp
import argparse

import pylab as pl
from plancklens import utils

sys.path.insert(0, '../params')
import params as parfile       

def qmap2dmap(qmap,nside):
    q2d = lambda l: (l*(l + 1)) ** 0.5 # potential -> deflection
    qlm = hp.map2alm(qmap)
    dmap = hp.alm2map(hp.almxfl(qlm,q2d(np.arange(len(qlm)))),nside)
    return dmap

def cli(cl):
    """Pseudo-inverse for positive cl-arrays.

    """
    ret = np.zeros_like(cl)
    ret[np.where(cl != 0)] = 1. / cl[np.where(cl != 0)]
    return ret   

def view_map(m, title, min=None, max=None, cmap='YlGnBu_r'):
     """ View map.
     """
     # TODO beautify this plot
     rot = [180, 60, 0]


     m = hp.read_map(m, verbose=False) if isinstance(m, str) else m
     m[ m==0. ] = np.nan # in case the input map is an apodization mask

     if min==None: min = m[ ~np.isnan(m) ].min()
     if max==None: max = m[ ~np.isnan(m) ].max()

     hp.orthview(m, title=title, min=min, max=max, rot=rot, half_sky=True, cmap=cmap)
     hp.graticule()
     import matplotlib.pyplot as plt
     plt.savefig(f'{title}.png', dpi=300)


parser = argparse.ArgumentParser(description='Check rec maps for AliCPT Lensing')   #创建一个解析器对象，用于定义和解析命令行参数。
parser.add_argument('-np', dest='np', type=int, default=1, help='Number of processes')   #添加要解析的命令行参数的规则  # -np 传递进程数
args = parser.parse_args()   #解析命令行参数，并返回一个包含解析结果的对象 args



qlm_p_mf = parfile.qlms_dd.get_sim_qlm_mf('p', parfile.mc_sims_mf_dd)
qresp = parfile.qresp_dd.get_response('p', 'p')
mask = hp.read_map('/sharefs/alicpt/users/chenwz/reconstruction/mask/masks/AliCPT_20uKcut150_C_2048.fits')

nside = 2048

def check_and_plot(nside, idx):
    q2k = lambda l : l * (l + 1) / 2
    lmax = parfile.qlms_dd.get_lmax_qlm('p')

    pmap_input = hp.read_map(f'/sharefs/alicpt/users/chenwz/reconstruction_2048_simons/test/ALILENS/sims/cmbs/map_P_2048_{idx+200:04d}.fits')
    qlm = parfile.qlms_dd.get_sim_qlm('p', idx )
    pmap_rec = hp.alm2map(hp.almxfl(qlm-qlm_p_mf, cli(qresp) ),nside=2048)  
    #pmap_mf = hp.alm2map(hp.almxfl(qlm_p_mf, cli(qresp) ),nside=2048) 



    dmap_input = qmap2dmap(pmap_input,nside)
    dmap_rec = qmap2dmap(pmap_rec,nside)

    #通过截取l范围可知，重建得到的phi在l=8到l=100左右的范围内是相对准确的，这与SNR或者功率谱的计算结果是一致的

    import curvedsky
    def map_cut(nside,map,lcut,lmax=None):
        if lmax == None:
            lmax = 3*nside-1
        blm = curvedsky.utils.hp_map2alm(nside,lmax,lmax,map)
        map_cut = curvedsky.utils.hp_alm2map(nside,lcut,lcut,blm[:lcut+1,:lcut+1])
        return map_cut
    llow = 20
    lcut = 2000


    dmap_rec_cut = map_cut(nside,dmap_rec,lcut,lmax=None) - map_cut(nside,dmap_rec,llow,lmax=None)
    dmap_input_cut = map_cut(nside,dmap_input,lcut,lmax=None) - map_cut(nside,dmap_input,llow,lmax=None)

    #view_map(dmap_input_cut * mask, title=' Input lensing map ', cmap='YlGnBu_r', min=-0.0024, max=0.0024)  
    #view_map(dmap_rec_cut * mask, title=' Reconstructed lensing map ',  cmap='YlGnBu_r', min=-0.0024, max=0.0024)
    view_map((dmap_input_cut - dmap_rec_cut) * mask, title=f' Difference lensing map {idx+200:04d}',  cmap='YlGnBu_r', min=-0.0024, max=0.0024)

seeds =  [_ for _ in range(0,121)]

#check_and_plot(0,nside)

if __name__ == '__main__':  #如果脚本被直接执行（而不是被导入为模块），则进入if __name__ == '__main__':代码块。
    # MultiProcessing
    pool = mp.Pool(processes=args.np)  #创建一个进程池（mp.Pool），并指定最大进程数为args.np，即通过命令行参数传递的进程数。
    for seed in seeds:      #使用for循环遍历名为seeds的迭代器中的每个元素，其中seeds可能是一个包含随机种子的列表或迭代器。
        pool.apply_async(check_and_plot, args=(nside,),
                kwds={'idx':seed})

    pool.close()  #执行完所有的异步任务后，使用pool.close()关闭进程池
    pool.join()   #使用pool.join()等待所有进程完成任务
