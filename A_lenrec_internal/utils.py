""" Utils 0_0
"""
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from plancklens.utils import cli



uKamin2uKpix = lambda n, npix : n / np.sqrt((360 * 60) ** 2 / np.pi / npix)  # 1° = pi/180 rad, 1' (arcmin) = pi/180/60 rad, 见PLANCK2015 (A.7)
uKpix2uKamin = lambda n, npix : n * np.sqrt((360 * 60) ** 2 / np.pi / npix)



def bl(fwhm, lmax=None, nside=None, pixwin=True):    #bl包括相当于beam window function 和 pixel window function,相当于e^-(l(l+1)* \sigma^2)(这是beam window function)
    """ Transfer function.

        * fwhm      : beam fwhm in arcmin
        * lmax      : lmax
        * pixwin    : whether include pixwin in beam transfer function
        * nside     : nside
    """
    assert lmax or nside
    lmax = min( 3 * nside - 1, lmax ) if nside and lmax else lmax if lmax else 3*nside - 1
    ret = hp.gauss_beam(fwhm * np.pi / 60. / 180., lmax=lmax)   #return: beam window function
    if pixwin:
        assert nside is not None
        ret *= hp.pixwin(nside, lmax=lmax)      #hp.pixwin: Return the pixel window function for the given nside
    return ret


def bl_eft(nrms_f, fwhm_f, lmax=None, pixwin=True, ret_nlev=False):  #bl_eft 包括noise_level、beam window function 和 pixel window function,即bl乘了(\Delta_t/p)^2
    """ Effective beam.
    """
    nrms_f = [ hp.read_map(nrms) if isinstance(nrms, str) else nrms for nrms in nrms_f ] #variance map 里存的不是variance,而是standard deviation(标准差)
    nside = hp.npix2nside(len(nrms_f[0]))

    nlev_f = np.array([ uKpix2uKamin(np.mean(nrms[nrms > 0] ** -2) ** -0.5 ,     #非mask区域对应的nrms(N功率谱方均根),对应1/(nlev)^2 = np.mean(1/(nrms)^2)                                                    
                        hp.nside2npix(nside)) for nrms in nrms_f]) # in uK.radians  #这里np.mean(nrms[nrms > 0] ** -2) ** -0.5是对高斯分布求平均(先平方避免正负相消)的算法，建议使用之前画出nrms的直方图(纵轴即个数)，看看是否符合高斯分布
    nlev = sum(nlev_f ** -2) ** -0.5 # in uk.arcmin
    bl_f = [ bl(fwhm, pixwin=pixwin, lmax=lmax, nside=nside) for fwhm in fwhm_f ]
    bl_eft = (sum([ nlev ** -2 * bl ** 2 for nlev, bl in zip(nlev_f, bl_f) ])
                * nlev ** 2) ** 0.5
    
    
    if ret_nlev:
        return nlev, bl_eft  #返回noiselevel和Effective beam
    else:
        return bl_eft        #仅返回Effective beam
        



def nl(nlev, fwhm, lmax=None, nside=None, pixwin=True): #cl|_noise = (nlev)^2 *exp(l(l+1)*\sigma^2/8/ln2)
    """ Detector noise spectrum

        * nlev      : noise level in uK.arcmin
    """
    # uK.arcmin -> uK.radians
    return ( (nlev * np.pi /60. /180.) * \
            cli(bl(fwhm, lmax=lmax, nside=nside, pixwin=pixwin)) ) ** 2   


def apodize_mask(m, savePath=None):  #见pymaster更佳：https://namaster.readthedocs.io/en/latest/sample_masks.html
    """ Apodization.
    平滑化mask边缘,由于边缘是0-1阶梯突变、非连续会导致:
    1、突变或不连续性的边缘效应可能导致分析中的伪像或异常结果。通过对边缘进行处理,可以减小这些效应。
    2、某些数值算法可能对不连续性或突变非常敏感,这可能导致数值不稳定性。通过对边缘进行处理,可以提高数值稳定性。
    3、地图通常代表某个物理系统或区域的属性,而物理系统通常不会在边缘处突变。因此,通过对边缘进行处理,可以使地图更符合物理一致性。
        * m         : noise rms map
    """
    # TODO more types of apodizaiton
    mask = np.zeros_like(m)
    mask[m!=0] = m[m!=0] ** -2 * len(m[m>0]) / sum( m[m>0] ** -2 )  ##################没看懂#####################

    if savePath:
        hp.write_map(savePath, mask, overwrite=True)
        
    return mask


def apodize_mask_new(m, savePath=None, case='C1'):  #参考pymaster而写：https://namaster.readthedocs.io/en/latest/sample_masks.html
    import healpy as hp
    import pymaster as nmt
    ''' 
    apomask, 边缘是cosine、sine或Gaussian函数
    
    C1 and C2: in these cases, pixels are multiplied by a factor f(with 0<=f<=1) based on their distance to the nearest fully masked pixel. 
               The choices of f in each case are documented in Section 3.4 of the C API documentation. 
               All pixels separated from any masked pixel by more than the apodization scale are left untouched.

    Smooth: in this case, all pixels closer to a masked pixel than 2.5 times the apodization scale are initially set to zero. 
            The resultingmap is then smoothed with a Gaussian kernel with standarddeviation given by the apodization scale. 
            Finally, all pixels originally masked are forced back to zero.
    '''
    # Apodization scale in degrees
    aposcale=2
    
    assert case == None or case == 'C1' or case == 'C2' or case == 'Smooth','check the case you choose: only for ''C1'', ''C2'' or ''Smooth'''
    if case is None or case == 'C1':
        apomask = nmt.mask_apodization(m, aposcale, apotype="C1")
    elif case == 'C2':
        apomask = nmt.mask_apodization(m, aposcale, apotype="C2")
    elif case == 'Smooth':
        apomask = nmt.mask_apodization(m, aposcale, apotype="Smooth")

    if savePath:
        hp.write_map(savePath, apomask, overwrite=True)
        
    return apomask
    





def view_map(m, title=None, savePath=None, min=None, max=None, cmap='YlGnBu_r'):
    """ View map.看和存map
    """
    # TODO beautify this plot
    rot = [180, 60, 0]


    m = hp.read_map(m, verbose=False) if isinstance(m, str) else m
    # FIXME DONT change the input map

    if min==None: min = m[ ~np.isnan(m) ].min()
    if max==None: max = m[ ~np.isnan(m) ].max()

    hp.orthview(m, title=title, min=min, max=max, rot=rot, half_sky=True, cmap=cmap)
    hp.graticule()
    plt.savefig(savePath, dpi=300)



def wl_f(nrms_f, fwhm_f, fwhm_c, pixwin=True, nl_c=False):
    """ Combination weights wl for each freq. channel.
        Substitute of SMICA weights in vmaps2vmap.

        对于模拟数据来说结合两个相差很多的channel对于SNR改进不大,对于实际观测数据今后可能有用。这里似乎未完工。

        * nrms_f    : noise rms maps, <nn>**0.5
        * fwhm_f    : in arcmin
        * fwhm_c    : in arcmin
        * pixwin    : pixwin
        * nl_c      : if True, return combined noise power spectrum

        NOTE
        没能够成功把不同频段的图结合起来,因为 variance map 还是有一些问题。
    """
    assert len(nrms_f) == len(fwhm_f)
    nrms_f = [ hp.read_map(nrms) if isinstance(nrms, str) else nrms for nrms in nrms_f ]
    nside = hp.npix2nside(len(nrms_f[0]))

    # [95GHz, 150GHz]
    # [11.35047730920254, 17.11698785862552]
    nlev_f = [ uKpix2uKamin(np.mean(nrms[nrms > 0] ** -2) ** -0.5 , hp.nside2npix(nside)) 
               for nrms in nrms_f]
    bl_f = [ bl(fwhm, pixwin=pixwin, nside=nside) for fwhm in fwhm_f ]   #######################################bl_f和bl_c分别是啥
    bl_c = bl(fwhm_c, pixwin=pixwin, nside=nside)
    nli_f = [ cli(nl(nlev, fwhm, nside=nside, pixwin=pixwin)) # inverse of nl
              for nlev, fwhm in zip(nlev_f, fwhm_f) ]
    wl_f = np.array([ cli(sum(nli_f)) * cli(bl) * bl_c * nli for bl, nli in zip(bl_f, nli_f) ])

    if nl_c:
        return wl_f, cli(sum(nli_f))
    else:
        return wl_f

def matrixshow(matrix,savePath=None):
    '''
    画出协方差矩阵
    '''
    import matplotlib.pyplot as plt
    import numpy as np

    data = np.around(matrix, decimals=2)

    # 创建一个图形
    fig, ax = plt.subplots()

    # 绘制热图并添加数值标签
    cax = ax.matshow(data, cmap='viridis')  # 使用'viridis'颜色映射

    # 循环遍历单元格
    for i in range(data.shape[0]):
        for j in range(i+1):    #协方差矩阵对称，这里画了其下三角部分；
        #for j in range(data.shape[1]): #画全部
            cell_value = data[i, j]
            text_color = 'black' if cell_value < 5 else 'white'  # 根据单元格值设定字体颜色
            ax.text(j, i, str(cell_value), va='center', ha='center', fontsize=15, color=text_color)

    # 设置x轴和y轴的刻度标签
    ax.set_xticks(np.arange(data.shape[0]))
    ax.set_yticks(np.arange(data.shape[1]))
    ax.set_xticklabels([i for i in range(data.shape[0])])
    ax.set_yticklabels([i for i in range(data.shape[1])])

    # 显示颜色条
    plt.colorbar(cax)

    # 隐藏x轴和y轴刻度线
    plt.tick_params(axis='both', which='both', length=0)

    # 调整字体颜色以适应背景颜色亮度,设置字体大小
    def adjust_text():
        for text in ax.texts:
            x, y = text.get_position()
            r, g, b, _ = cax.cmap(cax.norm(data[int(y), int(x)]))
            brightness = 0.299 * r + 0.587 * g + 0.114 * b  # 计算亮度
            if brightness > 0.5:
                text.set_color('black')
                text.set_size('small')
            else:
                text.set_color('white')
                text.set_size('small')

    adjust_text()  # 调整字体颜色

    fig.savefig(savePath, dpi=300)


    '''cmap可选颜色
    'viridis'：从紫色到黄色的顺序渐变。
    'plasma'：从紫色到橙色的顺序渐变，但亮度更高。
    'inferno'：从紫色到橙色的顺序渐变，但对比度更高。
    'magma'：从紫色到橙色的顺序渐变，对比度较高，亮度适中。
    'cividis'：一种颜色盲安全的颜色映射，适合可视化。
    'cool'：从蓝色到紫色的渐变。
    'hot'：从黑色到红色的渐变。
    'jet'：著名的彩虹颜色映射，从蓝色到红色，通常应该谨慎使用，因为它不太适合数据可视化。
    '''
        