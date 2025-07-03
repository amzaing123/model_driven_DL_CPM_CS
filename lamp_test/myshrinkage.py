import tensorflow as tf
import math
import numpy as np

pi = math.pi

####收缩函数

# def shrink_soft_threshold(r, rvar, theta):
#     if len(theta.get_shape()) > 0 and theta.get_shape() != (1,):
#         lam = theta[0] * tf.sqrt(rvar)  # 形状 [500, 1]，类型 float64
#         scale = theta[1]
#     else:
#         lam = theta * tf.sqrt(rvar)
#         scale = None
#
#     lam = tf.expand_dims(lam, axis=1)  # 扩展为 [500, 1, 1]
#     lam = tf.tile(lam, [1, 16, 1])  # 扩展为 [500, 16, 1]，类型 float64
#
#     lam = tf.maximum(lam, 0)  # 类型 float64
#     arml = tf.abs(r) - lam  # 类型 float64，形状 [500, 16, 1]
#
#     # 计算 max{|r_i| - θ_0 σ, 0}，并转换为 complex128
#     max_term = tf.maximum(arml, 0)  # 类型 float64
#     max_term_complex = tf.cast(max_term, tf.complex128)  # 转换为 complex128（实部为原值，虚部为 0）
#
#     # 计算 xhat
#     xhat = tf.sign(r) * max_term_complex  # tf.sign(r) 是 complex128，xhat 也是 complex128
#
#     # dxdr 的计算需要调整，因为它需要浮点类型
#     dxdr = tf.reduce_mean(tf.cast(tf.cast(arml, tf.float32) > 0, tf.float32), axis=0)
#
#     if scale is not None:
#         # scale 是 float64，转换为 complex128
#         scale_complex = tf.cast(scale, tf.complex128)
#         xhat = xhat * scale_complex  # 保持 complex128 类型
#         dxdr1 = dxdr * tf.cast(scale, tf.float32)  # dxdr 保持浮点类型
#         dxdr =tf.cast(dxdr1, tf.complex128)
#     return (xhat, dxdr)




###第三次
# def shrink_soft_threshold(r, rvar, theta):
#     if len(theta.get_shape()) > 0 and theta.get_shape() != (1,):
#         lam = theta[0] * tf.sqrt(rvar)  # 形状 [batch_size, 1]，类型 float64
#         scale = theta[1]
#     else:
#         lam = theta * tf.sqrt(rvar)  # 形状 [batch_size, 1]
#         scale = None
#
#     lam = tf.expand_dims(lam, axis=1)  # 扩展为 [batch_size, 1, 1]
#     lam = tf.tile(lam, [1, 16, 1])  # 扩展为 [batch_size, 16, 1]，类型 float64
#
#     lam = tf.maximum(lam, 0)  # 类型 float64
#     arml = tf.abs(r) - lam  # 类型 float64，形状 [batch_size, 16, 1]
#
#     # 计算 max{|r_i| - θ_0 σ, 0}，并转换为 complex128
#     max_term = tf.maximum(arml, 0)  # 类型 float64
#     max_term_complex = tf.cast(max_term, tf.complex128)  # 转换为 complex128
#
#     # 计算 xhat
#     xhat = tf.sign(r) * max_term_complex  # xhat 为 complex128
#
#     # 计算 dxdr，降维到 (batch_size,)
#     dxdr = tf.reduce_mean(tf.cast(tf.cast(arml, tf.float32) > 0, tf.float32), axis=[1, 2])  # 形状 [batch_size]
#     dxdr = tf.squeeze(dxdr, axis=-1)  # 确保形状为 (batch_size,)
#
#     if scale is not None:
#         scale_complex = tf.cast(scale, tf.complex128)
#         xhat = xhat * scale_complex
#         dxdr1 = dxdr * tf.cast(scale, tf.float32)  # 形状 (batch_size,)
#         dxdr = tf.cast(dxdr1, tf.complex128)  # 形状 (batch_size,), 类型 complex128
#
#     return (xhat, dxdr)

# def shrink_soft_threshold(r, rvar, theta):  ####gemini
#     """
#     修正后的软阈值收缩函数。
#
#     Args:
#         r (tf.Tensor): 输入张量，形状为 (?, 16, 1)，类型为 complex128。
#         rvar (tf.Tensor): r 的方差，形状为 (?, 1)，类型为 float64。
#         theta (tf.Tensor): 阈值参数，形状为 (2, 1)，类型为 float64_ref。
#
#     Returns:
#         tuple:
#             - xhat (tf.Tensor): 收缩后的估计值，形状和类型与 r 相同。
#             - dxdr (tf.Tensor): 导数的平均值，形状为 (?,)，类型为 complex128。
#     """
#     # 1. 解析 theta 参数
#     if len(theta.get_shape()) > 0 and theta.get_shape() != (1,):
#         lam_multiplier = theta[0]  # 阈值乘数
#         scale = theta[1]           # 缩放因子
#     else:
#         lam_multiplier = theta
#         scale = None
#
#     # 2. 计算阈值 lam
#     # tf.sqrt(rvar) 的形状是 (?, 1)
#     lam = lam_multiplier * tf.sqrt(rvar)  # lam 形状: (?, 1), dtype: float64
#
#     # 扩展 lam 以匹配 r 的形状 (?, 16, 1)
#     # (?, 1) -> (?, 1, 1) -> (?, 16, 1)
#     lam = tf.expand_dims(lam, axis=1)
#     lam = tf.tile(lam, [1, 16, 1])
#     lam = tf.maximum(lam, 0.0)
#
#     # 3. 应用软阈值公式计算 xhat
#     # arml = |r| - lam
#     arml = tf.abs(r) - lam  # 形状: (?, 16, 1), dtype: float64
#
#     # max_term = max(|r| - lam, 0)
#     max_term = tf.maximum(arml, 0.0)
#
#     # xhat = sign(r) * max_term
#     # 注意：tf.sign(complex) 返回单位复数，这是正确的
#     xhat = tf.sign(r) * tf.cast(max_term, tf.complex128) # 形状: (?, 16, 1), dtype: complex128
#
#     # 4. 计算导数 dxdr
#     # dxdr 是一个指示函数 I(|r| > lam) 的平均值
#     # arml > 0 等价于 |r| > lam
#     is_over_threshold = tf.cast(arml > 0, tf.float32) # 形状: (?, 16, 1), dtype: float32
#
#     # *** 核心修正 ***
#     # 沿着维度 1 和 2 (16 和 1) 求平均，保留维度 0 (?)
#     dxdr = tf.reduce_mean(is_over_threshold, axis=[1, 2]) # 形状: (?,), dtype: float32
#
#     # 5. 如果有 scale，应用缩放因子
#     if scale is not None:
#         scale_float = tf.cast(scale, tf.float32)
#         scale_complex = tf.cast(scale, tf.complex128)
#
#         # 对 xhat 和 dxdr 应用 scale
#         xhat = xhat * scale_complex
#         dxdr = dxdr * scale_float # dxdr 保持浮点类型
#
#     # 6. 将 dxdr 转换为 complex128 以匹配函数签名中的描述
#     # 根据您的要求，最终返回 complex128 类型
#     dxdr_complex = tf.cast(dxdr, tf.complex128)
#
#     # 最终返回的 dxdr_complex 形状为 (?,)，类型为 complex128，符合您的理论预期
#     return (xhat, dxdr_complex)


def shrink_soft_threshold(r, rvar, theta):
    if len(theta.get_shape()) > 0 and theta.get_shape() != (1,): ###可用grok
        lam = theta[0] * tf.sqrt(rvar)  # 形状 [NONE, 1]，类型 float64
        scale = theta[1]
    else:
        lam = theta * tf.sqrt(rvar)
        scale = None

    lam = tf.expand_dims(lam, axis=1)  # 扩展为 [NONE, 1, 1]
    lam = tf.tile(lam, [1, 16, 1])  # 扩展为 [NONE, M, 1]

    lam = tf.maximum(lam, 0)  # 类型 float64
    arml = tf.abs(r) - lam  # 类型 float64，形状 [NONE, M, 1]

    # 转换为 complex128 类型
    max_term = tf.maximum(arml, 0)
    max_term_complex = tf.cast(max_term, tf.complex128)

    # 计算 xhat
    xhat = tf.sign(r) * max_term_complex  # xhat 是 complex128 类型

    # 计算 dxdr，沿 axis=[1, 2] 减少，保留批量维度
    dxdr = tf.reduce_mean(tf.cast(tf.cast(arml, tf.float32) > 0, tf.complex128), axis=[1, 2])  # 形状 [NONE,]

    if scale is not None:
        scale_complex = tf.cast(scale, tf.complex128)
        xhat = xhat * scale_complex
        dxdr = dxdr * tf.cast(scale, tf.complex128)  # dxdr 保持 complex128 类型

    return (xhat, dxdr)
def shrink_MMV(r, x_gain, tau, L):
    with tf.device('/cpu:0'):
        G = int(r.shape[1])
        K = int(r.shape[2])
        lam = L / G
        delta = 1 / tau - 1 / (tau + 1)
        filter_gain = tf.cast(
            1 / (1 + tau) / (1 + (1 - lam) / lam * tf.exp(K * (tf.log(1 + 1 / tau) - delta * x_gain))),
            tf.complex128)
        a = tf.matrix_diag(filter_gain)
        # b = tf.matrix_diag(tf.transpose(filter_gain, (1, 0)))
        xhat = tf.matmul(a, r)
        # xhat = tf.transpose(xhat, (1, 2, 0))
        return xhat, filter_gain


def shrink_bgest(r, rvar, K, theta):
    with tf.device('/cpu:0'):
        xvar1 = abs(theta[0, ...])
        loglam = theta[1, ...]  # log(1/lambda - 1)
        beta = 1 / (1 + rvar / xvar1)
        r_gain = tf.reduce_sum(tf.square(tf.abs(r)), axis=2)
        r2scale = r_gain * beta / rvar
        rho = tf.exp(loglam + .5 * K * tf.log(1 + xvar1 / rvar) - .5 * r2scale)
        rho1 = rho + 1
        gain = tf.cast(beta / rho1, tf.complex128)
        xhat = tf.matmul(tf.matrix_diag(gain), r)         # r为(None,M,K)
        # dxdr = beta * ((1 + rho * (1 + r2scale)) / tf.square(rho1))
        dxdr = tf.reduce_mean(gain, axis=1)
        return (xhat, dxdr)


def laplace_shrinkage(r, rvar, theta):  ###存在问题
    with tf.device('/cpu:0'):
        """
        实现拉普拉斯分布下的收缩函数，参考伯努利-高斯收缩函数的结构。
        支持实数和复数输入：实数使用 tf.multiply，复数使用矩阵运算。

        参数：
        r: 输入张量，形状 (None, M, K)，复数（complex128）
        rvar: 方差 (sigma)，形状 (None, 1)，实数（float64）

        theta: 参数 b，标量，实数（float64）

        返回：
        xhat: 收缩后的值，形状 (None, M, K)复数（complex128）
        dxdr: xhat 相对于 r 的导数，形状 (M,)复数（complex128）
        """
        rvar1 = tf.square(rvar)
        rvar2 = rvar1 + theta
        rvar3 = tf.divide(1, rvar2)
        rvar4 = rvar1 * rvar3
        rvar5 = tf.complex(rvar4, tf.constant(0.0, dtype=tf.float64))
        dxdr = rvar5
        r1 = tf.transpose(r, [0, 2, 1])  # 形状: [500, 1, 16]
        dxdr1 = tf.expand_dims(dxdr, axis=2)  # 形状: [500, 1, 1]
        xhat = tf.matmul(dxdr1, r1)  # 形状: [500, 1, 16]
        xhat = tf.transpose(xhat, [0, 2, 1])  # 形状: [500,16 ,1 ]
        return xhat, dxdr

def pwlin_grid(r_,rvar_,theta_,dtheta = .75):
    """piecewise linear with noise-adaptive grid spacing.
    returns xhat,dxdr
    where
        q = r/dtheta/sqrt(rvar)
        xhat = r * interp(q,theta)

    all but the  last dimensions of theta must broadcast to r_
    e.g. r.shape = (500,1000) is compatible with theta.shape=(500,1,7)
    """
    ntheta = int(theta_.get_shape()[-1])
    scale_ = dtheta / tf.sqrt(rvar_)
    ars_ = tf.clip_by_value( tf.expand_dims( tf.abs(r_)*scale_,-1),0.0, ntheta-1.0 )
    centers_ = tf.constant( np.arange(ntheta),dtype=tf.float32 )
    outer_distance_ = tf.maximum(0., 1.0-tf.abs(ars_ - centers_) ) # new dimension for distance to closest bin centers (or center)
    gain_ = tf.reduce_sum( theta_ * outer_distance_,axis=-1) # apply the gain (learnable)
    xhat_ = gain_ * r_
    dxdr_ = tf.gradients(xhat_,r_)[0]
    return (xhat_,dxdr_)
def get_shrinkage_function(name):
    try:
        return {
            'bg': (shrink_bgest, [[1.0], [math.log(1/.1-1)]]),
            'MMV': (shrink_MMV, [1.0]),
            'soft': (shrink_soft_threshold, [[1.0], [1.0]]),
            'lap':(laplace_shrinkage,[1.0]),
            'pwgrid': (pwlin_grid, np.linspace(.1, 1, 15).astype(np.float32)),
        }[name]
    except KeyError as ke:
        raise ValueError('unrecognized shrink function %s' % name)
        sys.exit(1)