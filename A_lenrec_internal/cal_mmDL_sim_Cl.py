import healpy as hp
import numpy as np


for idx in range(100):
    map_cmb = hp.read_map(f'/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_ILC/mmDL_maps_2048/{idx:05d}/lensedcmb_{idx:05d}.fits', field=(0,1,2))

    if idx == 0:
        cl = hp.anafast(map_cmb, lmax=3*2048-1)
    else:
        cl += hp.anafast(map_cmb, lmax=3*2048-1)

cl /= 100

L = np.arange(len(cl[0]))
fac = L * (L + 1) / (2 * np.pi)
TT, EE, BB, TE = cl[0]*fac, cl[1]*fac, cl[2]*fac, cl[3]*fac
# 合并为一个二维数组 (N行 × 5列)
data = np.column_stack([L, TT, EE, BB, TE])

# 输出文件名
outfile = 'mmDL_lensedCls.dat'

# 写入文件
with open(outfile, "w") as f:
    # 写表头
    f.write("#    L    TT             EE             BB             TE\n")
    # 写数据行
    for row in data:
        f.write(f"{int(row[0]):6d}    {row[1]:.5E}    {row[2]:.5E}    {row[3]:.5E}    {row[4]:.5E}\n")

print(f"✅ 已保存到 {outfile}")

    