#the qe keys you want for the output of lensing reconstruction
lib_qe_keys={
#            'ptt':['royalblue', 'TT'] # Temperature only
           'p_p':['green', 'Pol'] # Polarization only
#           ,'p':['tomato', 'MV']# MV estimator
          } 


ALILENS = "/sharefs/alicpt/users/chenwz/reconstruction_multi_process2/process3/ALILENS" #the library directory of your dataset and the final output

ALILENS_cmb = "/sharefs/alicpt/users/chenwz/reconstruction_2048_simons/test/ALILENS/sims/cmbs" #the library directory of your dataset and the final output

ALILENS_noise = "/sharefs/alicpt/users/chenwz/reconstruction_2048_simons/test/ALILENS/sims/noise" #the library directory of your dataset and the final output

ALILENS_fg = "/sharefs/alicpt/users/chenwz/Testarea/FORECAST/ALL_MAP_2048/FG/145"

nside = 2048
lmax = 3*nside-1
dlmax = 4096
seeds = [_ for _ in range(200,321)]  #0到200共201个realization (即simulation)
nlev = [#"/home/jkhan/2021/data/Noise_ALI_IHEP_20200730_48/I_NOISE_95_C_1024.fits",   #variance map 里存的不是variance,而是standard deviation(标准差)
#"/sharefs/alicpt/users/chenwz/reconstruction/mask/4/I_NOISE_150_C_2048.fits"
#"/sharefs/alicpt/users/chenwz/reconstruction_2048_simons/I_noise_Simons_2048.fits"
"/sharefs/alicpt/users/chenwz/reconstruction_2048_simons/I_noise_Simons_2048_inhomo_145.fits"
        ]
fwhm_f = [#19.
         #11.
        #3.
        1.4
        ]


#library that you store your mask
mask_bb="/sharefs/alicpt/users/chenwz/reconstruction/mask/masks/AliCPT_UNPfg_filled_C_2048.fits" #binary mask (只有0-1)
mask_apodiz="/sharefs/alicpt/users/chenwz/reconstruction/mask/masks/mask_2048_Sm.fits" # apodized binary mask (边缘是cosine、sine或Gaussian函数)


#library that you store your 48 module noise data
#####lib_path_48="/sharefs/alicpt/users/chenwz/download/alilens/2022/sims/noise_48/" #total noise of 48 module
#####lib_cov_48="/sharefs/alicpt/users/chenwz/download/alilens/2022/sims/conv_48/" #total noise saved for covariance matrix calculation

bias=120 # how many sets of CMB maps used to calculate mean field  #必须是偶数，因为要计算两个平场，44 sets 平分。且平分之后最好也是偶数，否则可能引入bias.
var=0 # how many sets of CMB maps used to calculate covariance matrix
nset=10 # the number of group  
nwidth=12 # the width of each group (how many sets in each group)  #注：每组内进行shuffle, 即让 j=i+1,且cyclically，见params.py 的 ss_dict
#p.s. as the sets in bias will be to calculate 2 mean field, I recommand that the group width equals to one half of bias.
#必须满足：(1) nset * nwidth = bias + var (2) nwidth <= bias/2 且 nwidth <= var, 即计算每个平场和协方差矩阵不能跨组

#library that you store your teb maps as well as your final recontruction results:

#the library directory storing detection noise residue maps: (noise realization maps(而上面的I_NOISE_150_C_1024.fits是noise std map),已经过ILC等处理，扣除了一部分noise)
lib_cls_res='/disk1/home/hanjk/2022/runs-48/noise_residuals/TEnilc-Bcilc_proj-noise_11arcmin_sim%d.fits' 

#the library directory storing total residue maps which is the summation of the detection noise residue maps and foreground residue maps(已经过ILC扣除foreground): 
lib_cls_noise='/disk1/home/hanjk/2022/runs-48/total_residuals/tot-residual_TEnilc-Bcilc_11arcmin_sim%d.fits' #noise realization maps(包括fg residue了,以后可能还包括systematic error)

#the noise maps which are used to caculate inverse variance (ivfs):
lib_cls_con='/disk1/home/hanjk/2022/runs-48/total_residuals/tot-residual_TEnilc-Bcilc_11arcmin_sim%d.fits'
#lib_cls_con='/disk1/home/hanjk/2022/runs/total_residuals/tot-residual_TEnilc-Bcilc_11arcmin_sim%02d.fits'

#the one set of teb noise map to be used in delensing estimation
lib_Ali_map_noise="/disk1/home/hanjk/2022/runs-48/total_residuals/tot-residual_TEnilc-Bcilc_11arcmin_sim198.fits"
lib_Ali_map_res="/disk1/home/hanjk/2022/runs-48/noise_residuals/TEnilc-Bcilc_proj-noise_11arcmin_sim198.fits"
