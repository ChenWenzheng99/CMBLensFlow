#load modules
print('loading modules...')
import numpy as np
import basic
import pickle
import curvedsky
#import cmb
import healpy as hp
import pymaster as nmt
import matplotlib.pyplot as plt

##########################################################################

import os,sys
import pylab as pl
import numpy as np
import lenspyx
import healpy as hp

# sys.path.append('/sharefs/alicpt/users/chenwz/download/cmblensplus2/utils')
# sys.path.append('/sharefs/alicpt/users/chenwz/Testarea/FORECAST/Foreground/foreground_run')


camb_path = os.path.realpath(os.path.join(os.getcwd(),'..'))    
sys.path.insert(0,camb_path)



import camb
from camb import model, initialpower ,correlations

from scipy.special import factorial

import pymaster as nm


from utils_mine import *


def g_function(x):
    # 将x转为numpy数组以便矢量化操作
    x = np.asarray(x)

    # 创建输出数组
    out = np.empty_like(x, dtype=float)

    # 定义一些mask
    mask_one = np.isclose(x, 1.0, atol=1e-14)  # 接近1的元素
    mask_nonpos = (x <= 0)  # 非正值元素

    # 对接近1的值，g(x)=0
    out[mask_one] = 0.0

    # 对非正值的元素，根据需要设为np.nan或其他处理
    out[mask_nonpos] = np.nan

    # 对剩余正常值应用公式
    mask_main = ~(mask_one | mask_nonpos)
    x_main = x[mask_main]

    # 应用公式：sign(x-1)* sqrt[2*(x - ln(x) - 1)]
    out[mask_main] = np.sign(x_main - 1.0) * np.sqrt(2.0*(x_main - np.log(x_main) - 1.0))

    return out


def compute_U_and_eigvals(A, B, ell_len, epsilon=1e-12):
    """
    对于每个ell:
    1. 对A_l特征分解，求得 A^{-1/2}.
    2. 计算 C = A^{-1/2} B A^{-1/2}.
    3. 对C进行特征分解，得到特征向量U和特征值。

    参数：
    A, B: 形状为 (n, n, ell_len) 的协方差矩阵。
    ell_len: ell数量
    epsilon: 正则项用于提高数值稳定性。

    返回：
    U: (n, n, ell_len) 每个ell对应的特征向量矩阵
    eigvals: (n, ell_len) 每个ell对应的特征值向量
    """
    n = A.shape[0]
    U = np.zeros((n, n, ell_len))
    eigvals_array = np.zeros((n, ell_len))

    for l in range(ell_len):
        # 对 A_l 特征分解
        w_A, v_A = np.linalg.eigh(A[:, :, l])
        # 防止数值不稳定，对过小特征值加正则项
        w_A_reg = np.maximum(w_A, epsilon)
        A_inv_sqrt = v_A @ np.diag(1.0 / np.sqrt(w_A_reg)) @ v_A.T

        # C = A^{-1/2} B A^{-1/2}
        C = A_inv_sqrt @ B[:, :, l] @ A_inv_sqrt

        # 对 C 特征分解
        w_C, v_C = np.linalg.eigh(C)
        U[:, :, l] = v_C
        eigvals_array[:, l] = w_C

    return U, eigvals_array

def get_Cgl(C, Chat, Cfhalf, lmax, epsilon=1e-12):
    """
    使用HL-likelihood近似转换 C 与 Chat 为 Cgl。
    
    参数：
    C, Chat, Cfhalf: 形状 (n, n, lmax) 的协方差矩阵
    lmax: ell数量
    epsilon: 正则项防止数值不稳定
    
    返回：
    Cgl: (n, n, lmax) 经过HL变换后的矩阵
    """
    from collections.abc import Sequence

    if isinstance(C, Sequence):
        C_final = np.zeros((1, 1, lmax))
        for l in range(lmax):
            rat = Chat / C
            C_final = g_function(rat) * (Cfhalf**2)
        return C_final
    
    if C.shape[0] == 1:
        C_final = np.zeros((1, 1, lmax))
        for l in range(lmax):
            rat = Chat[0,0,l] / C[0,0,l]
            C_final[0,0,l] = g_function(rat) * (Cfhalf[0,0,l]**2)
        return C_final
    
    else:
        n = C.shape[0]

        # 从 C 与 Chat 中提取特征值与特征向量
        U, eigvals = compute_U_and_eigvals(C, Chat, lmax, epsilon=epsilon)

        # 对特征值应用HL变换
        # eigvals: (n, lmax)
        transformed_vals = g_function(eigvals)  # 对整个矩阵元素应用
        
        Cgl = np.zeros((n, n, lmax))
        for l in range(lmax):
            # g(D_l) 对角矩阵
            G = U[:, :, l] @ np.diag(transformed_vals[:, l]) @ U[:, :, l].T
            Cgl[:, :, l] = Cfhalf[:, :, l] @ G @ Cfhalf[:, :, l].T

        return Cgl


"""
def g_function(x):
    # HL-likelihood g函数
    # 处理x接近1的情况避免数值问题
    if np.isclose(x, 1.0, atol=1e-14):
        return 0.0
    if x <= 0:
        # x必须>0，否则log(x)无意义
        return np.nan
    return np.sign(x - 1.0) * np.sqrt(2.0 * (x - np.log(x) - 1.0))


def g_function(x):
    # HL-likelihood g函数
    # 处理x接近1的情况避免数值问题
    return np.sign(x - 1.0) * np.sqrt(2.0 * (x - np.log(x) - 1.0))

def transform0(Cf, Chat, Cfhalf, lmax, epsilon=1e-12):
    
    #对协方差矩阵进行HL-likelihood近似下的变换。

    #参数：
    #- Cf: fiducial协方差矩阵, 形状 (n, n, lmax)
    #- Chat: 观测协方差矩阵, 形状 (n, n, lmax)
    #- Cfhalf: fiducial协方差的平方根, 形状 (n, n, lmax)
    #- lmax: 多极矩数量
    #- epsilon: 正则化参数，当Cfhalf接近奇异时提供稳定性

    #返回：
    #- C_final: 变换后的矩阵 (n, n, lmax)
    
    n = Cf.shape[0]

    # 特殊情况：1x1矩阵
    if n == 1:
        C_final = np.zeros((1, 1, lmax))
        for l in range(lmax):
            rat = Chat[0,0,l] / Cf[0,0,l]
            C_final[0,0,l] = g_function(rat) * (Cfhalf[0,0,l]**2)
        return C_final

    C_final = np.zeros_like(Cf)

    for l in range(lmax):
        # 若Cfhalf接近奇异，可加正则项
        Cfhalf_reg = Cfhalf[:,:,l] + epsilon * np.eye(n)
        Cf_inv_half = np.linalg.inv(Cfhalf_reg)

        # X_l = Cf^{-1/2} Chat_l Cf^{-1/2}
        X_l = Cf_inv_half @ Chat[:,:,l] @ Cf_inv_half.T

        # 特征分解 X_l
        w, U = np.linalg.eigh(X_l)

        # 对特征值w应用g函数
        w_g = np.array([g_function(x) for x in w])

        # 构建g(D)
        G = U @ np.diag(w_g) @ U.T

        # 映射回原空间: C_g,l = Cfhalf * G * Cfhalf^T
        C_final[:,:,l] = Cfhalf_reg @ G @ Cfhalf_reg.T

    return C_final
"""


#Some transformation used for HL likelihood

def transform1(C, Chat, Cfhalf, lmax):   ###############   Also correct one, equivalent to 'get_Cgl' (have passed check)
    # HL transformation of the matrices
    if C.shape[0] == 1:
        # Special case for 1x1 matrix
        rat = Chat[0, 0] / C[0, 0]
        C[0, 0] = (np.sign(rat - 1) *
                   np.sqrt(2 * np.maximum(0, rat - np.log(rat) - 1)) *
                   Cfhalf[0, 0] ** 2)
        C_final = C.copy()
        return C_final
    
    # Initialize C_final with the same shape as C
    C_final = np.zeros_like(C)

    for l in range(0, lmax):  # 11 is the number of ell bins, should be changed if the number of ell bins is changed.
        #print(l)
        # Eigenvalue decomposition 
        diag, U = np.linalg.eigh(C[:,:,l])  # Unormalizaed, U is the eigenvector matrix, diag is the eigenvalues D (Eq.43 下方)
        
        # Rotate Chat in the eigenvector space of C
        rot = U.T.dot(Chat[:,:,l]).dot(U)
        
        # Normalize the rotated matrix
        roots = np.sqrt(diag)
        for i, root in enumerate(roots):
            rot[i, :] /= root
            rot[:, i] /= root
        
        # Project the normalized matrix back
        rot = U.dot(rot).dot(U.T)
        
        # Eigenvalue decomposition of the normalized matrix
        diag, rot = np.linalg.eigh(rot)
        
        # HL transformation of the eigenvalues
        diag = np.sign(diag - 1) * np.sqrt(2 * np.maximum(0, diag - np.log(diag) - 1))  #This corresponds to g(D) in Eq.44

        # Construct the final transformed matrix (Equivalently to codes below)
        #U = np.dot(Cfhalf[:,:,l], rot)
        #for i, d in enumerate(diag):
        #    rot[:, i] = U[:, i] * d
        #C_final[:,:,l] = np.dot(rot, U.T)

        # Construct the final transformed matrix
        rot = U.dot(np.diag(diag)).dot(U.T)
        C_final[:,:,l] = Cfhalf[:,:,l].dot(rot).dot(Cfhalf[:,:,l]) 
    
    return C_final


"""
def matrix_C2X(C):
    
    #Generalized function to extract specific elements from a symmetric matrix C.
    #This function can handle any square matrix C and will return a flattened array of the upper triangular elements.

    #Parameters:
    #    C (ndarray): A symmetric matrix of shape (n, n, ell_len).

    #Returns:
    #    X (ndarray): A 1D array containing the upper triangular elements of C.
    
    n = C.shape[0]  # Get the dimension of the square matrix
    ell_len = C.shape[2]  # Assuming the matrix is 3D with the third dimension as ell_len
    
    # Initialize an empty list to store the upper triangular elements
    X_list = []

    # Loop over the upper triangular indices and append the corresponding elements to the list
    for i in range(n):
        for j in range(i, n):  # Ensure we only get upper triangular elements (including the diagonal)
            X_list.append(C[i, j, :])
    
    # Convert the list to a numpy array
    X = np.array(X_list)
    
    return X
"""
# Example usage
#C = np.array(C_mo)  # Example of a 7x7 symmetric matrix with 3 components
#X = matrix_C2X(C)
#print(X)


def matrix_C2X(M, triangle="upper"):
    """
    Extract the upper or lower triangular (including diagonal) elements from a symmetric 3D matrix.

    Parameters
    ----------
    M : ndarray
        A symmetric 3D matrix of shape (n, n, ell_len).
    triangle : str, optional
        Specifies which triangular part to extract: 'upper' for upper triangular (default),
        or 'lower' for lower triangular.

    Returns
    -------
    X : ndarray
        A 1D array containing the specified triangular elements of M for all ell.
        The elements are flattened in the order they are extracted.
    """
    n = M.shape[0]
    if triangle == "upper":
        idx = np.triu_indices(n)  # Upper triangular indices
    elif triangle == "lower":
        idx = np.tril_indices(n)  # Lower triangular indices
    else:
        raise ValueError("Invalid option for 'triangle'. Use 'upper' or 'lower'.")
    return M[idx]


"""
def matrix_vec2C(data_vec, tri=None):
    
    #根据 data_vec 中的上三角元素生成对称矩阵 C。
    #参数:
    #    data_vec (list or array): 包含上三角部分（包括对角线）的元素。
    #返回:
    #    C (ndarray): 重构后的对称矩阵。
    
    # 计算矩阵的维度 N
    length = len(data_vec)
    n = int((-1 + np.sqrt(1 + 8 * length)) / 2)  # 计算出矩阵的维度
    ell_len = len(data_vec[0])  # ell 的长度
    
    if n * (n + 1) // 2 != length:
        raise ValueError(f"data_vec 的长度 {length} 不匹配对称矩阵的上三角元素数目")

    # 初始化 N x N 的矩阵
    C = np.zeros((n, n, ell_len))
    
    if tri == 'upper':
        # 将 data_vec 的元素填入上三角矩阵
        index = 0
        for i in range(n):
            for j in range(i, n):
                C[i, j, :] = data_vec[index]
                C[j, i, :] = data_vec[index]  # 利用对称性填充下三角部分
                index += 1
    elif tri == 'lower':
        # 将 data_vec 的元素填入下三角矩阵
        index = 0
        for i in range(n):
            for j in range(i + 1):  # 下三角，包括对角线
                C[i, j, :] = data_vec[index]
                C[j, i, :] = data_vec[index]  # 利用对称性填充上三角部分
                index += 1
    else:
        raise ValueError("tri must be either 'upper' or 'lower'")
    return C
"""

def matrix_vec2C(data_vec, tri='upper'):
    """
    从包含上三角（或下三角）元素的一维数据向量中重构对称矩阵 C。

    参数:
    ----------
    data_vec : array_like
        包含对称矩阵上三角（含对角线）或下三角（含对角线）元素的二维数组。
        data_vec 的形状应为 (M, ell_len)，其中 M = n*(n+1)/2。
    tri : str, optional
        指定 data_vec 中的数据是上三角还是下三角元素。
        可选值为 'upper' 或 'lower'，默认 'upper'。

    返回:
    ----------
    C : ndarray
        重构后的对称三维矩阵，形状为 (n, n, ell_len)。
        其中 n 由 data_vec 的长度推断得到。
    """
    data_vec = np.array(data_vec)
    length = data_vec.shape[0]
    ell_len = data_vec.shape[1]

    # 根据长度求 n
    # n(n+1)/2 = length
    # n = (-1 + sqrt(1+8*length))/2
    n = int((-1 + np.sqrt(1 + 8 * length)) // 2)

    if n * (n + 1) // 2 != length:
        raise ValueError(f"data_vec 的长度 {length} 不匹配对称矩阵的上三角元素数目")

    C = np.zeros((n, n, ell_len))

    if tri == 'upper':
        # 使用上三角索引
        i_inds, j_inds = np.triu_indices(n)
    elif tri == 'lower':
        # 使用下三角索引
        i_inds, j_inds = np.tril_indices(n)
    else:
        raise ValueError("tri must be either 'upper' or 'lower'")

    # 将 data_vec 中的元素依次赋值给 C 的对称位置
    for idx, (i, j) in enumerate(zip(i_inds, j_inds)):
        C[i, j, :] = data_vec[idx]
        C[j, i, :] = data_vec[idx]

    return C



from camb.mathutils import chi_squared as _chi2
from ctypes import c_int

def fast_chi_squared(covinv, x):
    if len(x) != covinv.shape[0] or covinv.shape[0] != covinv.shape[1]:
        raise ValueError('Wrong shape in chi_squared')
    x = np.ascontiguousarray(x)
    covinv = np.ascontiguousarray(covinv)
    return _chi2(covinv, x)





"""
############################ Another code to solve C_gl #############################
def compute_U_and_D(A, B, ell_len):
    # 初始化U和D的输出矩阵
    n = A.shape[0]  # Get the dimension of the square matrix
    vec_len = n   # Number of elements in the upper triangular part
    #ell_len = C.shape[2]  # Assuming the matrix is 3D with the third dimension as ell_len
    U = np.zeros((vec_len, vec_len, ell_len))
    D = np.zeros((vec_len, vec_len, ell_len))
    
    for l in range(ell_len):
        # 计算A的特征值和特征向量
        eigvals_A, eigvecs_A = np.linalg.eigh(A[:, :, l])
        
        # 计算A的逆平方根
        A_inv_sqrt = eigvecs_A @ np.diag(1.0 / np.sqrt(eigvals_A)) @ eigvecs_A.T
        
        # 计算A^-1/2 * B * A^-1/2
        C = A_inv_sqrt @ B[:, :, l] @ A_inv_sqrt
        
        # 对C进行特征值分解
        eigvals_C, eigvecs_C = np.linalg.eigh(C)
        
        # U是C的特征向量矩阵，D是C的特征值对角矩阵
        U[:, :, l] = eigvecs_C
        D[:, :, l] = np.diag(eigvals_C)
    
    return U, D

def get_Cgl(C, Chat, Cfhalf, lmax):
    U, D = compute_U_and_D(C, Chat, lmax)

    n = C.shape[0]  # Get the dimension of the square matrix
    vec_len = n  # Number of elements in the upper triangular part
    ell_len = lmax
    # HL transformation of the eigenvalues
    diag = np.sign(D - 1) * np.sqrt(2 * np.maximum(0, D - np.log(D) - 1))  #This corresponds to g(D) in Eq.44

    Cgl = np.zeros((vec_len, vec_len, ell_len))
    rot = np.zeros((vec_len, vec_len, ell_len))
    for l in range(ell_len):
        # Construct the final transformed matrix
        rot[:, :, l] = U[:, :, l].dot(np.diag(diag[:, :, l])).dot(U[:, :, l].T)
        Cgl[:, :, l] = Cfhalf[:, :, l].dot(rot[:, :, l]).dot(Cfhalf[:, :, l].T)
    return Cgl
"""

def compute_covmat(vec, lmax, lmin=2, lbin_scale=1):
    """
    Calculate the covariance matrix (COVMAT) for the given vec dictionary containing power spectra data.

    Parameters:
    - vec: dict, where keys are tuples of pairs and values are arrays representing power spectra.
    - lmax: int, maximum ℓ to compute.
    - lmin: int, minimum ℓ (default is 2).
    - lbin_scale: int, scale factor for the ℓ bin (default is 1).

    Returns:
    - COV_th: np.ndarray, shape (num_ell_bins, len(vec), len(vec)), covariance matrix.
    """
    # Determine the available length for each vector in vec to prevent out-of-bounds errors
    max_length = min([len(v) for v in vec.values()])
    lmax = min(lmax, max_length - 1)  # Adjust lmax to be within available data range

    # Initialize covariance matrix
    num_ell_bins = lmax - lmin + 1
    cov_shape = (num_ell_bins, len(vec), len(vec))
    COV_th = np.zeros(cov_shape)
    
    for idx, l in enumerate(range(0, lmax + 1)):
        for i, key1 in enumerate(vec.keys()):
            for j, key2 in enumerate(vec.keys()):
                # Determine the correct keys for covariance calculation
                key3 = (key1[0], key2[0]) if key1[0] <= key2[0] else (key2[0], key1[0])
                key4 = (key1[1], key2[1]) if key1[1] <= key2[1] else (key2[1], key1[1])
                key5 = (key1[0], key2[1]) if key1[0] <= key2[1] else (key2[1], key1[0])
                key6 = (key1[1], key2[0]) if key1[1] <= key2[0] else (key2[0], key1[1])
                
                # Calculate covariance matrix element
                if lbin_scale == 1:
                    l_eff = l
                    COV_th[idx, i, j] = 1 / (2 * l_eff + 1) * (vec[key3][l] * vec[key4][l] + vec[key5][l] * vec[key6][l])
                else:
                    l_eff = ( (l + 0.5) * lbin_scale - 0.5) * lbin_scale  # Effective ℓ for binned power spectra l_bin ≈ delta_bin * ((2 * idx + 1) * delta_bin - 1) / 2
                    COV_th[idx, i, j] = 1 / l_eff * (vec[key3][l] * vec[key4][l] + vec[key5][l] * vec[key6][l])
    
    return COV_th

def plot_matrix_with_log(matrix, cmap='viridis', title='2D Matrix Visualization with Logarithmic Values and Grid'):
    """
    绘制带有取对数值和网格线的二维矩阵。
    
    参数:
    matrix (np.array): 输入的二维矩阵。
    cmap (str): 颜色映射方案。默认值为 'viridis'。
    title (str): 图像标题。默认值为 '2D Matrix Visualization with Logarithmic Values and Grid'。
    """
    
    # 对矩阵中的每个元素取自然对数
    log_matrix = np.log(matrix)
    
    # 使用 Matplotlib 绘制取对数后的矩阵
    fig, ax = plt.subplots()
    cax = ax.matshow(log_matrix, cmap=cmap)
    
    # 添加颜色条
    fig.colorbar(cax)
    
    # 在每个格子中显示取对数后的数值
    rows, cols = log_matrix.shape
    #for i in range(rows):
    #    for j in range(cols):
    #        ax.text(j, i, f'{log_matrix[i, j]:.2f}', ha='center', va='center', color='white')
    
    # 添加内线
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    
    # 隐藏主刻度线
    ax.tick_params(which='major', bottom=False, left=False)
    
    # 添加标题
    plt.title(title)
    
    # 显示图像
    plt.show()


def covmat_inv(COV):
    """
    计算协方差矩阵的逆矩阵。
    
    参数:
    COV (np.ndarray): 协方差矩阵，形状为 (n, m, m)。
    
    返回:
    COV_inv (np.ndarray): 协方差矩阵的逆矩阵，形状为 (n, m, m)。
    若某个矩阵为奇异矩阵，其逆矩阵将被设置为 0 矩阵。
    """
    COV_inv = np.zeros_like(COV)  # 初始化为 0 矩阵，防止异常情况时未定义的值
    for i in range(COV.shape[0]):
        try:
            COV_inv[i, :, :] = np.linalg.inv(COV[i, :, :])
        except np.linalg.LinAlgError:
            # 奇异矩阵时，保持为零矩阵
            COV_inv[i, :, :] = np.zeros_like(COV[i, :, :])
            print(f"Matrix at ell={i} is singular and has been set to zero.")
    return COV_inv


def covmat_corr_inv(COV):
    """
    计算协方差矩阵的逆矩阵。
    
    参数:
    COV (np.ndarray): 协方差矩阵，形状为 (n, m, m)。
    
    返回:
    COV_inv (np.ndarray): 协方差矩阵的逆矩阵，形状为 (n, m, m)。
    若某个矩阵为奇异矩阵，其逆矩阵将被设置为 0 矩阵。
    """
    COV_inv = np.zeros_like(COV)  # 初始化为 0 矩阵，防止异常情况时未定义的值
    for i in range(COV.shape[0]):
        for j in range(COV.shape[0]):
            try:
                    COV_inv[i, j, :, :] = np.linalg.inv(COV[i, j, :, :])
            except np.linalg.LinAlgError:
                    # 奇异矩阵时，保持为零矩阵
                    COV_inv[i, j, :, :] = np.zeros_like(COV[i, j, :, :])
                    print(f"Matrix at ell={i} is singular and has been set to zero.")
    return COV_inv


from scipy.linalg import sqrtm
def matrix_sqrt(matrix):
    """
    Compute the square root of a matrix.
    
    Parameters:
    - matrix: np.ndarray, input matrix.
    
    Returns:
    - sqrt_matrix: np.ndarray, square root of the input matrix.
    """
    ell_len = matrix.shape[2]
    matrix_root = matrix.copy()
    for i in range(ell_len):
        matrix_root[:, :, i] = sqrtm(matrix[:, :, i])
    return matrix_root