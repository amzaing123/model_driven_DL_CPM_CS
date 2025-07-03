import tensorflow as tf
import numpy as np
import myshrinkage1
import myshrinkage2
import sys
from scipy.io import loadmat, savemat
import math
pi = math.pi
###添加以下代码
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6 #根据你的需求调整
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# def gen_noise(y, snr):
#     """
#     Generate complex Gaussian white noise for a given signal and SNR using TensorFlow.
#
#     Parameters:
#     y (tf.Tensor): Input signal (complex-valued, arbitrary shape).
#     snr (float): Signal-to-noise ratio in dB.
#
#     Returns:
#     n (tf.Tensor): Complex Gaussian white noise with the same shape as y.
#     """
#     # Ensure y is complex128
#     tf.set_random_seed(42)
#     y = tf.cast(y, tf.complex128)
#
#     # Calculate signal power
#     signal_power = tf.reduce_mean(tf.square(tf.abs(y)))
#
#     # Calculate noise power based on SNR
#     snr_linear = 10 ** (snr / 10)
#     noise_power = signal_power / snr_linear
#
#     # Noise variance (real and imaginary parts each contribute half)
#     noise_var = noise_power / 2
#
#     # Generate complex Gaussian noise
#     noise_real = tf.random.normal(shape=tf.shape(y), mean=0.0, stddev=1.0, dtype=tf.float64)
#     noise_imag = tf.random.normal(shape=tf.shape(y), mean=0.0, stddev=1.0, dtype=tf.float64)
#     noise = tf.complex(noise_real, noise_imag)
#
#     # Scale noise to desired variance
#     n = tf.sqrt(tf.cast(noise_var, tf.complex128)) * noise
#
#     return n
def gen_noise(y, snr):  # 定义一个函数，用于生成噪声
    ypower = tf.reduce_mean(tf.square(tf.abs(y)), axis=0, keepdims=True)  # 计算信号 y 的平均功率
    noise_var = tf.cast(ypower / ((10 ** (snr / 10)) * 2), tf.complex128)  # 计算噪声方差，转换为复数类型
    noise = tf.complex(real=tf.random_normal(shape=tf.shape(y), dtype=tf.float64),  # 生成服从标准正态分布的噪声，实部
                       imag=tf.random_normal(shape=tf.shape(y), dtype=tf.float64))  # 虚部
    n = tf.sqrt(noise_var) * noise  # 计算噪声

    return n  # 返回噪声
def batch_pseudo_inverse(Y, rcond=1e-15):
    s, u, v = tf.linalg.svd(Y)
    threshold = tf.reduce_max(s, axis=-1, keepdims=True) * rcond
    s_inv = tf.where(tf.greater(s, threshold),
                     tf.reciprocal(s),
                     tf.zeros_like(s))
    s_inv_diag = tf.linalg.diag(s_inv)
    # 获取 u 或 v 的复数类型，以确保类型一致
    complex_dtype = u.dtype
    # 将实数张量 s_inv_diag 转换为复数类型
    s_inv_diag_complex = tf.cast(s_inv_diag, dtype=complex_dtype)
    Y_pinv = tf.matmul(v, tf.matmul(s_inv_diag_complex, tf.transpose(u, perm=[0, 2, 1])))
    return Y_pinv


def clip_complex_dxdr(dxdr_, eps):
    """
    对复数输入 dxdr_ 的模进行裁剪，通过比较模大小选择对应的复数。

    参数:
        dxdr_: 输入张量，复数类型（tf.complex64 或 tf.complex128）
        eps: 复数，裁剪边界（标量或与 dxdr_ 形状兼容）

    返回:
        裁剪后的复数张量，根据模比较选择 dxdr_、one - eps 或 eps
    """
    # 定义 one = 1.0 + 0j
    one = tf.complex(tf.constant(1.0, dtype=tf.float64), tf.constant(0.0, dtype=tf.float64))

    # 计算 one - eps（复数减法）
    one_minus_eps = one - eps

    # 计算模
    magnitude_dxdr = tf.abs(dxdr_)
    magnitude_one_minus_eps = tf.abs(one_minus_eps)
    magnitude_eps = tf.abs(eps)

    # 第一步：比较 |one - eps| 和 |dxdr_|
    # 如果 |one - eps| > |dxdr_|，选择 dxdr_，否则选择 one_minus_eps
    dxdr_1 = tf.where(
        magnitude_one_minus_eps > magnitude_dxdr,
        dxdr_,
        tf.ones_like(dxdr_) * one_minus_eps
    )


    # 第二步：比较 |dxdr_1| 和 |eps|
    # 如果 |dxdr_1| > |eps|，选择 dxdr_1，否则选择 eps
    magnitude_dxdr_1 = tf.abs(dxdr_1)
    dxdr_final = tf.where(
        magnitude_dxdr_1 > magnitude_eps,
        dxdr_1,
        tf.ones_like(dxdr_) * eps
    )

    return dxdr_final
###函数通过多层迭代和非线性收缩操作构建了一个自定义的 LAMP 网络，用于信号重建问题。每一层都进行了初始化和参数更新，最终返回构建的网络层次结构和占位符。
def build_LVAMP_dense(M, N, K,  snr, T, shrink):
    """ Builds the non-SVD (i.e. dense) parameterization of LVAMP
    and returns a list of trainable points(name,xhat_,newvars)
    """
    eta,theta_init = myshrinkage2.get_shrinkage_function(shrink)
    layers=[]
    var_all =[]
    A_ = tf.placeholder(tf.complex128, (M, N))  # 输入A ###注意输入格式
    h_ = tf.placeholder(tf.complex128, (None, N, K))  # 输入H ###注意输入格式
    h1 = tf.transpose(h_, [1, 0, 2])  # [N, None, K]
    h1 = tf.reshape(h1, (N, -1))  # [N, None*K]
    with tf.device('/cpu:0'):  # 将矩阵乘法放在 CPU 上
        ytemp1 = tf.matmul(A_, h1)  # ytemp1 的形状应为 [M, None*K] = [32, 2]
        noise = gen_noise(ytemp1, snr)
        ytemp1 = ytemp1 + noise
    ytemp1 = tf.reshape(ytemp1, (M, tf.shape(h_)[0], K))  # (M,NONE,K)
    ytemp_ = tf.transpose(ytemp1, [1, 0, 2])  # (NONE,M,K)
    y_ = ytemp_  # (NONE,M,K)  ###没问题

    with tf.device('/cpu:0'):
        y_pinv = batch_pseudo_inverse(y_)  # 兼容1.13.1版本 # (NONE,K,M)

    y_pinv=tf.reshape(y_pinv,(-1,M)) # (NONE*K,M)
    with tf.device('/cpu:0'):
        Hinit = tf.matmul(h1,y_pinv ) # 此处Hinit 为 X* ，输出[N,M]

    Hreal_ = tf.Variable(tf.real(Hinit), name='Hreal_' + str(snr) + '_1')
    var_all.append(Hreal_)
    Himag_ = tf.Variable(tf.imag(Hinit), name='Himag_' + str(snr) + '_1')
    var_all.append(Himag_)
    H_ = tf.complex(Hreal_, Himag_, name='H') #(N, M)

    y1=tf.transpose(y_,[1, 0, 2]) #(M,NONE,K)
    y1=tf.reshape(y1, (M, -1))#(M,NONE*K)
    with tf.device('/cpu:0'):
        xhat_init1 = tf.matmul(H_,y1) #(N, M)*(M,NONE*K)=(N,NONE*K)

    xhat_init2=tf.reshape(xhat_init1, (N, tf.shape(h_)[0], K))#(N,NONE,K)
    xhat_init=tf.transpose(xhat_init2,[1, 0, 2]) #(NONE,N,K)
    # print("xhat_init dtype:", xhat_init.dtype, "shape:", xhat_init.shape)
    # var_all.append( ('Linear',xhat_init,None) )

    # if shrink=='pwgrid':
    #     # theta_init = np.linspace(.01,.99,15).astype(np.float32)
    #     theta_init = tf.linspace(0.01, 0.99, 15)
    # vs_def = tf.constant(1, dtype=tf.float64)
    # if not iid:
    #     theta_init = np.tile( theta_init ,(N,1,1))
    #     vs_def = np.tile( vs_def ,(N,1))

    # 使用tf.tile扩展张量
    # theta_init = tf.tile(theta_init, [N, 1, 1])
    vs_def = tf.constant(1, dtype=tf.float64)  # 可能存在问题 #应该为复数，暂不确定

    # vs_def = tf.constant(1 + 0j, dtype=tf.complex128)# 可能存在问题 #应该为复数，暂不确定
    # vs_real = tf.Variable(tf.real(vs_def), name='vsreal_' + str(snr) + '_1')
    # var_all.append(vs_real)
    # vs_imag = tf.Variable(tf.real(vs_def), name='vsimag_' + str(snr) + '_1')
    # var_all.append(vs_imag)
    # print("vs_def dtype:", vs_def.dtype, "shape:", vs_def.shape)
    # vs_def = tf.tile(vs_def, [N, 1])
    # print(f"theta 类型: {type(theta_init)}")  # 在调用 eta 之前
    theta_ = tf.Variable(theta_init, dtype=tf.float64, name='theta_' + str(snr) + '_1')
    # vs_ = tf.complex(vs_real, vs_imag, name='vs')
    # vs_= tf.complex(vs_real, vs_imag, name='vs')  # (N,M)
    vs_ = tf.Variable(vs_def, name='vs_'+ str(snr) + '_1')
    var_all.append(theta_)
    var_all.append(vs_)

    rhat_ = xhat_init#(NONE,N,K)
    with tf.device('/cpu:0'):
        # y_conj = tf.math.conj(y_)  # 计算共轭
        # y_y_conj = y_ * y_conj  # 逐元素乘法：y * y^*
        # result = tf.reduce_sum(y_y_conj, axis=0) / N
        # # rvar_nl_ = vs_ * tf.reduce_sum(y_*y_,0)/N
        # vs_=tf.math.abs(vs_)
        # rvar_ = tf.multiply(vs_,tf.reduce_sum(tf.square(tf.abs(y_)), 0) / N) #float64  存在问题
        # rvar_=tf.multiply (vs_,tf.expand_dims(tf.reduce_sum(tf.square(tf.norm(tf.abs(y_), axis=[1, 2])), 1)/N))
        y_1=(tf.expand_dims(tf.square(tf.norm(tf.abs(y_), axis=[1, 2])), 1))/N
        # print("y_1 dtype:", y_1.dtype, "y_1 shape:", y_1.shape)
        rvar_=vs_*y_1 #float64 (none,k)

    # print("rhat_ dtype:", rhat_.dtype, "shape:", rhat_.shape)
    # print("rvar_ dtype:", rvar_.dtype, "shape:", rvar_.shape)

    # xhat_,alpha_nl_ = eta(r_,rvar_,K,theta_init )  #xhat_,complex128  alpha_nl_,complex128
    #rvar_格式出现错误，不应该是（16，1）应该为（none，1）
    xhat_,dxdr_ = eta(rhat_,rvar_,K,theta_ )  #xhat_nl_,complex128  alpha_nl_,complex128 # xhat_ (None, M, K)  bg
    # xhat_,dxdr_ = eta(rhat_,rvar_,theta_ )  #xhat_nl_,complex128  alpha_nl_,complex128 # xhat_ (None, M, K)  soft
    # print("xhat_ dtype:", xhat_.dtype, "shape:", xhat_.shape)
    # print("dxdr_ dtype:", dxdr_.dtype, "shape:", dxdr_.shape)
    layers.append( ('LVAMP-{0} T={1}'.format(shrink,1),xhat_, tuple(var_all), tuple(var_all), (0,)) )


    for t in range(2,T+1):
        ###  rhat 网络上边
        dxdr_ = tf.reduce_mean(dxdr_, axis=0)  # each col average dxdr
        # print("dxdr_ dtype:", dxdr_.dtype, "shape:", dxdr_.shape)
        one = tf.complex(tf.constant(1.0, dtype=tf.float64), tf.constant(0.0, dtype=tf.float64))
        # with tf.device('/cpu:0'):
        #     gain_r = tf.divide(one, one - dxdr_) # dxdr_ ()
        #     dxdr_r = dxdr_ * rhat_
        #     rhat_ = gain_r * (xhat_ - dxdr_r)

        with tf.device('/cpu:0'):
            gain_r = tf.divide(one, one - dxdr_) # dxdr_ ()
        dxdr_r = dxdr_ * rhat_
        rhat_ = gain_r * (xhat_ - dxdr_r)

        # print("rhat_0 dtype:", rhat_.dtype, "rhat_0 shape:", rhat_.shape)
        ###rvar 网络下边
        dxdr_ = tf.math.abs(dxdr_)
        # with tf.device('/cpu:0'):
        #     gain_v1 = tf.divide(1, 1 - dxdr_)
        #     gain_v2 = gain_v1 * dxdr_
        #     gain_v = tf.sqrt(gain_v2)
        #     rvar_ = rvar_ * gain_v
        with tf.device('/cpu:0'):
            gain_v1 = tf.divide(1, 1 - dxdr_)
        gain_v2 = gain_v1 * dxdr_
        gain_v = tf.sqrt(gain_v2)
        rvar_ = rvar_ * gain_v

        # print("rvar_ dtype:", rvar_.dtype, "rvar_ shape:", rvar_.shape)
        # gain_r = tf.cast(gain_r, tf.complex128)
        # gain_v_abs = tf.math.abs(gain_v)
        # print("gain_r dtype:", gain_r.dtype, "shape:", gain_r.shape)

            #rhat
            # dxdr_r = dxdr_*rhat_

            # rhat_ = tf.multiply(gain_r, (xhat_ - dxdr_r)) # xhat_ (None, M, K)
            # xhat_dxdr=xhat_ - dxdr_r
             # xhat_ (None, M, K)

            # rvar_ = tf.multiply(rvar_, gain_v_abs)

        Hreal_ = tf.Variable(tf.real(Hinit), name='Hreal_' + str(snr) + '_' + str(t))
        var_all.append(Hreal_)
        Himag_ = tf.Variable(tf.imag(Hinit), name='Himag_' + str(snr) + '_' + str(t))
        var_all.append(Himag_)
        H_ = tf.complex(Hreal_, Himag_, name='H')
        G_init=0.9 * tf.eye(N, dtype=tf.complex128)
        G_real = tf.Variable(tf.real(G_init),dtype=tf.float64,name='Greal_' + str(snr) + '_' +str(t))
        var_all.append(G_real)
        G_imag = tf.Variable(tf.imag(G_init),dtype=tf.float64,name='Gimag_' + str(snr) + '_' +str(t))
        var_all.append(G_imag)
        G_ = tf.complex(G_real, G_imag, name='G')  # (N,M)
        rhat_=tf.transpose(rhat_, [1,0,2])# rhat_ (M,None, K)
        rhat_=tf.reshape(rhat_, (M, -1))# rhat_ (M,None*K)

        y_1 = tf.transpose(y_, [1, 0, 2])  # y_ (M,None, K)
        y_2 = tf.reshape(y_1, (M, -1))  # y_ (M,None*K)
        # print("y_2 dtype:", y_2.dtype, "y_2 shape:", y_2.shape)
        with (tf.device('/cpu:0')):
            xhat_1 = tf.matmul(H_,y_2)   # H (N,M) * y_2(M,NONE*K) xhat_1 (N,NONE*K)
            # print("xhat_1 dtype:", xhat_1.dtype, " shape:", xhat_1.shape)
            xhat_2 = tf.matmul(G_,rhat_)#(N,M) * rhat_(M,NONE*K) xhat_2 (N,NONE*K)
            # print("G_ dtype:", G_.dtype, " shape:", G_.shape)
            # print("xhat_2 dtype:", xhat_2.dtype, "xhat_2 shape:", xhat_2.shape)
            # print("rhat_ dtype:", rhat_.dtype, "rhat_ shape:", rhat_.shape)
            xhat_ =xhat_1+xhat_2  # xhat_(N,NONE*K)

        xhat_=tf.reshape(xhat_, (tf.shape(h_)[0],M,K))
        # print("xhat_ dtype:", xhat_.dtype, "xhat_ shape:", xhat_.shape)
        # layers.append( ('LVAMP-{0} lin T={1}'.format(shrink,1+t),xhat_, tuple(var_all), tuple(var_all), (0,) ) )
        dxdr_ = tf.expand_dims(tf.diag_part(G_),1)
        # dxdr_ = tf.reduce_mean(dxdr_, axis=0)
        #存在问题
        # 对复数张量 dxdr_ 求模
        real_part = tf.cast(.5 / N, dtype=tf.float64)
        imag_part = tf.cast(0, dtype=tf.float64)
        eps = tf.complex(real_part, imag_part)
        # print("eps dtype:", eps.dtype, "eps shape:", eps.shape)
        # eps = tf.cast(.5/N, dtype=tf.float64)
        # print("dxdr_1 dtype:", dxdr_.dtype, "dxdr_1 shape:", dxdr_.shape)
        dxdr_=clip_complex_dxdr(dxdr_,eps) #比较大小确定阈值
        # print("dxdr_2 dtype:", dxdr_.dtype, "dxdr_2 shape:", dxdr_.shape)
        # 存在问题

        # vs_real = tf.Variable(tf.real(vs_def), name='vsreal_' + str(snr) + 't')
        # var_all.append(vs_real)
        # vs_imag = tf.Variable(tf.real(vs_def), name='vsimag_' + str(snr) + 't')
        # vs_= tf.complex(vs_real, vs_imag, name='vs')  # (N,M)
        vs_= tf.Variable(vs_def, name='vs_'+ str(snr)+ '_'+str(t), dtype=tf.float64)
        var_all.append(vs_)
        ###rhat 网络上层
        dxdr_ = tf.reduce_mean(dxdr_, axis=0)  # each col average dxdr  #新添
        one1 = tf.complex(tf.constant(1.0, dtype=tf.float64), tf.constant(0.0, dtype=tf.float64))
        # dxdr_=tf.math.abs(dxdr_)
        # with tf.device('/cpu:0'):
        #     gain_r = tf.divide(one1, one1 - dxdr_) # 1-vt  dxdr_(1,)
        #     # print("rhat_ dtype:", rhat_.dtype, "rhat_ shape:", rhat_.shape)
        #     # print("gain_r dtype:", gain_r.dtype, "gain_r shape:", gain_r.shape)
        #     # print("dxdr_ dtype:", dxdr_.dtype, "dxdr_ shape:", dxdr_.shape)
        #     dxdr_r1 = dxdr_ * rhat_  # dxdr_r(16, ?)  rhat_ shape: (16, ?)
        #     dxdr_r2=tf.reshape(dxdr_r1,(M, tf.shape(h_)[0], K)) #(M,None,K)
        #     dxdr_r=tf.transpose(dxdr_r2, [1, 0, 2])#(None,M,K)
        #     # print("dxdr_r dtype:", dxdr_r.dtype, "dxdr_r shape:", dxdr_r.shape)
        #     # print("xhat_ dtype:", xhat_.dtype, "xhat_ shape:", xhat_.shape)
        #     rhat_ = gain_r * (xhat_ - dxdr_r) #xhat_(None,M,K)   (?, 16, ?)现在不对
        #     # print("rhat_ dtype:", rhat_.dtype, "rhat_ shape:", rhat_.shape)

        with tf.device('/cpu:0'):
            gain_r = tf.divide(one1, one1 - dxdr_) # 1-vt  dxdr_(1,)
        # print("rhat_ dtype:", rhat_.dtype, "rhat_ shape:", rhat_.shape)
        # print("gain_r dtype:", gain_r.dtype, "gain_r shape:", gain_r.shape)
        # print("dxdr_ dtype:", dxdr_.dtype, "dxdr_ shape:", dxdr_.shape)
        dxdr_r1 = dxdr_ * rhat_  # dxdr_r(16, ?)  rhat_ shape: (16, ?)
        dxdr_r2=tf.reshape(dxdr_r1,(M, tf.shape(h_)[0], K)) #(M,None,K)
        dxdr_r=tf.transpose(dxdr_r2, [1, 0, 2])#(None,M,K)
        # print("dxdr_r dtype:", dxdr_r.dtype, "dxdr_r shape:", dxdr_r.shape)
        # print("xhat_ dtype:", xhat_.dtype, "xhat_ shape:", xhat_.shape)
        rhat_ = gain_r * (xhat_ - dxdr_r) #xhat_(None,M,K)   (?, 16, ?)现在不对
        # print("rhat_ dtype:", rhat_.dtype, "rhat_ shape:", rhat_.shape)

        #rvar 网络下层
        dxdr_ = tf.math.abs(dxdr_)
        # with tf.device('/cpu:0'):
        #     gain_v1 = tf.divide(1, 1 - dxdr_)
        #     gain_v2 = gain_v1 * dxdr_
        #     gain_v = tf.sqrt(gain_v2)
        #     gain_vs=gain_v*vs_
        #     rvar_ = rvar_ * gain_vs         #现在存在报错，维度不匹配

        with tf.device('/cpu:0'):
            gain_v1 = tf.divide(1, 1 - dxdr_)
        gain_v2 = gain_v1 * dxdr_
        gain_v = tf.sqrt(gain_v2)
        gain_vs=gain_v*vs_
        rvar_ = rvar_ * gain_vs         #现在存在报错，维度不匹配

        # print("rhat_2 dtype:", rhat_.dtype, "rhat_2 shape:", rhat_.shape)
        rhat_=tf.reshape(rhat_,(M, tf.shape(h_)[0], K))
        rhat_ = tf.transpose(rhat_, [1, 0, 2])  # (NONE,M,K)
        # y1 = tf.reshape(y1, (M, -1))  # (M,NONE*K)
        theta_ = tf.Variable(theta_init,name='theta_'+ str(snr)+ '_'+str(t),dtype=tf.float64)
        var_all.append(theta_)
        # print("rhat_ dtype:", rhat_.dtype, "rhat_ shape:", rhat_.shape)
        xhat_,dxdr_ = eta(rhat_,rvar_,K,theta_ )#rvar_ float64   rhat_ complex128 (None,M,K) bg
        # xhat_,dxdr_ = eta(rhat_,rvar_,theta_ )#rvar_ float64   rhat_ complex128 (None,M,K) soft
        # print("xhat_ dtype:", xhat_.dtype, "xhat_ shape:", xhat_.shape)
        # print("dxdr_ dtype:", dxdr_.dtype, "dxdr_ shape:", dxdr_.shape)
        layers.append( ('LVAMP-{0} T={1}'.format(shrink,t),xhat_, tuple(var_all), tuple(var_all), (0,)) )

    return layers ,h_, A_
#函数通过遍历模型的每一层，计算损失函数 (NMSE)，定义优化器（如 Adam），并根据不同的学习率设置训练阶段。
# 最终返回一个包含所有训练阶段的列表，供后续训练过程中使用。通过这种方式，模型的参数可以逐层得到优化和更新
def setup_training(layers, x_, A, K, N, M, trinit=1e-3, refinements=(0.8, 0.5, 0.1)):
    """
    设置 LAMP 网络的训练阶段，确保每层训练基于前一层的参数。

    参数：
        layers: LAMP 网络层列表，来自 build_LAMP
        x_: 输入信号占位符，形状 [None, N, K]
        A: 输入矩阵占位符，形状 [M, N]
        K, N, M: 网络维度参数
        trinit: 初始学习率
        refinements: 精调学习率列表

    返回：
        training_stages: 训练阶段列表，每项包含 (名称, 输出, 损失, NMSE, 优化器, 变量列表, 所有变量, 标志)
    """
    training_stages = []
    print("layers:", layers)
    for i, layer in enumerate(layers):
        print(f"layer {i}: {layer}, type: {type(layer)}, len: {len(layer)}")
    for name, xhat_, var_list, var_all, flag in layers:
        with tf.device('/cpu:0'):  # 在 CPU 上执行矩阵操作
            # 计算估计信号 hhat_
            # print("xhat_ dtype:", xhat_.dtype, "xhat_ shape:", xhat_.shape)
            xhat1 = tf.transpose(xhat_, [1, 0, 2])  # [M, None, K]
            xhat2 = tf.reshape(xhat1, (M, -1))  # [M, None*K]
            hhat = xhat2  # 直接使用 xhat2，假设输出已为估计信号
            hhat = tf.reshape(hhat, (M, tf.shape(x_)[0], K))  # [M, None, K]
            hhat_ = tf.transpose(hhat, [1, 0, 2])  # [None, M, K]

            # 计算 NMSE 损失
            nmse_ = tf.reduce_mean(
                tf.square(tf.norm(tf.abs(hhat_ - x_), axis=[1, 2])) /
                tf.square(tf.norm(tf.abs(x_), axis=[1, 2]))
            )
            loss_ = nmse_
            # print(var_list)
            # 定义初始优化器
            train_ = tf.train.AdamOptimizer(trinit).minimize(loss_, var_list=var_list)

        print(f"训练变量: {[v.name for v in var_list]}")

        if var_list is not None:
            # 添加初始训练阶段
            training_stages.append((name, hhat_, loss_, nmse_, train_, var_list, var_all, flag))

        # 添加精调阶段
        for fm in refinements:
            with tf.device('/cpu:0'):
                train2_ = tf.train.AdamOptimizer(fm * trinit).minimize(loss_, var_list=var_all)
            training_stages.append((
                f"{name} trainrate={fm}", hhat_, loss_, nmse_, train2_, (), var_all, flag
            ))

    return training_stages


def load_trainable_vars(sess, filename):
    other = {}
    try:
        variables = tf.trainable_variables()
        tv = {str(v.name).replace(':', '_'): v for v in variables}
        d = loadmat(filename)
        for k, val in d.items():
            if k in tv:
                print(f"Restoring {k}, saved shape: {np.shape(val)}, variable shape: {tv[k].shape}")
                # 如果变量是标量且加载值是 [1,1]，转换为标量
                if tv[k].shape == () and np.shape(val) == (1, 1):
                    val = float(val[0, 0])
                # 如果变量是标量且加载值是 [1]，转换为标量
                elif tv[k].shape == () and np.shape(val) == (1,):
                    val = float(val[0])
                sess.run(tf.assign(tv[k], val))
            else:
                if k == 'done':
                    # 处理 done 列表中的字符串
                    val = [str(item).strip() for item in val.flatten()]
                other[k] = val
    except IOError:
        print(f"无法加载文件: {filename}")
    return other

def save_trainable_vars(sess, filename, snr, **kwargs):
    save = {}
    for v in tf.trainable_variables():
        name_parts = str(v.name).split('_')
        if len(name_parts) > 1 and name_parts[1] == str(snr):
            val = sess.run(v)
            # 如果是标量，转换为 Python 标量
            if val.shape == ():
                val = float(val)  # 或 val.item()
            save[str(v.name).replace(':', '_')] = val
    save.update(kwargs)
    savemat(filename, save)

def assign_trainable_vars(sess, var_list, var_list_old):
    # for i in range(len(var_list)):
    for i in range(len(var_list_old)):
        temp = sess.run(var_list_old[i])
        # print(temp)
        sess.run(tf.assign(var_list[i], temp))
# def assign_trainable_vars(sess, var_list, var_list_old):
#     min_len = min(len(var_list), len(var_list_old))
#     if min_len < len(var_list):
#         print(f"Warning: var_list_old has {len(var_list_old)} variables, but var_list needs {len(var_list)}. Initializing extra variables.")
#     for i in range(min_len):
#         temp = sess.run(var_list_old[i])
#         sess.run(tf.assign(var_list[i], temp))
#     if min_len < len(var_list):
#         sess.run(tf.variables_initializer(var_list[min_len:]))

#函数实现了一个完整的训练过程，包括加载数据、初始化会话、执行训练、计算验证误差、调整学习率和保存训练变量。
# 通过逐步优化不同的训练阶段，确保模型在每个阶段都得到充分的训练，最终返回包含所有状态的会话

def do_training(h_, training_stages, A, savefile, trainingfile, validationfile, snr, iv1=100, maxit=50000, better_wait=10000):
    """
    执行 LAMP 网络的训练，继承前一层的最优参数，保存最优 NMSE 参数和每层 NMSE。

    参数：
        h_: 输入信号占位符，形状 [None, N, K]
        training_stages: 训练阶段列表，来自 setup_training
        A: 输入矩阵占位符，形状 [M, N]
        savefile: 保存训练参数的文件路径
        trainingfile: 训练数据 .mat 文件路径
        validationfile: 验证数据 .mat 文件路径
        snr: 信噪比 (dB)
        iv1: 验证间隔
        maxit: 最大迭代次数
        better_wait: 早停等待迭代次数

    返回：
        sess: TensorFlow 会话
    """
    # 加载训练数据
    try:
        Dtraining = loadmat(trainingfile)
        ht = Dtraining['h_mc_train']  # 训练信号，形状 [N, K, trainingsize]
        A1 = Dtraining['st02']  # 训练矩阵 A，形状 [M, N]
        trainingsize = np.size(ht, axis=2)
    except KeyError as e:
        raise KeyError(f"训练数据文件中缺少键: {e}")

    # 加载验证数据
    with tf.device('/cpu:0'):
        try:
            Dvalidation = loadmat(validationfile)
            hv = Dvalidation['h_mc_val']  # 验证信号
            hv = np.transpose(hv, [2, 0, 1])  # 形状 [None, N, K]
        except KeyError as e:
            raise KeyError(f"验证数据文件中缺少键: {e}")

    # 选择一个批次数据作为 h_ 的初始值
    batch_size = 64
    rand_index = np.random.choice(trainingsize, size=batch_size, replace=False)
    h_init = ht[..., rand_index]  # 形状 [N, K, batch_size]
    h_init = np.transpose(h_init, [2, 0, 1])  # 形状 [batch_size, N, K]

    # 初始化 TensorFlow 会话
    sess = tf.Session()
    sess.run(tf.global_variables_initializer(), feed_dict={A: A1,h_: h_init})

    # 加载保存的状态
    state = load_trainable_vars(sess, savefile)
    done = state.get('done', [])
    # 确保 done 是 Python 列表
    if isinstance(done, np.ndarray):
        done = done.tolist()
    log = state.get('log', [])
    # 确保 log 是 Python 列表
    if isinstance(log, np.ndarray):
        log = log.tolist()
    layernmse = state.get('layernmse', [])
    # 确保 layernmse 是 Python 列表
    if isinstance(layernmse, np.ndarray):
        layernmse = layernmse.tolist()
    best_vars = state.get('best_vars', {})  # 存储每层最优参数

    # 用于参数继承
    prev_layer_vars = {}  # 存储前一层的最优参数 {变量名: 值}

    for name, xhat_, loss_, nmse_, train_, var_list, var_all, flag in training_stages:
        if name in done:
            print(f"已完成训练: {name}，跳过。")
            continue

        print(f"\n开始训练: {name}，变量: {','.join([v.name for v in var_list ])}")

        # 继承前一层的参数
        if prev_layer_vars and var_list:
            print("继承前一层的最优参数...")
            for v in var_list:
                if v.name in prev_layer_vars:  # 直接比较变量名（字符串）
                    try:
                        sess.run(tf.assign(v, prev_layer_vars[v.name]))
                        print(f"已为 {v.name} 赋值")
                    except Exception as e:
                        print(f"警告: 无法为 {v.name} 赋值，错误: {e}")

        nmse_history = []
        best_nmse = float('inf')
        best_var_values = {}

        for i in range(maxit + 1):
            if i % iv1 == 0:
                # 在验证集上计算 NMSE
                with tf.device('/cpu:0'):
                    nmse = sess.run(nmse_, feed_dict={h_: hv, A: A1})
                nmse = round(nmse, 5)
                if np.isnan(nmse):
                    raise RuntimeError(f"NMSE 为 NaN 在 {name}")

                nmse_history.append(nmse)
                nmse_dB = 10 * np.log10(nmse)
                nmsebest_dB = 10 * np.log10(min(nmse_history))

                # 保存当前最优参数
                if nmse < best_nmse:
                    best_nmse = nmse
                    best_var_values = {v.name: sess.run(v) for v in (var_list if var_list else var_all)}
                    print(f"更新最优 NMSE: {nmsebest_dB:.6f} dB 在迭代 {i}")

                sys.stdout.write(
                    f"\ri={i:<6d} NMSE={nmse_dB:.6f} dB (best={nmsebest_dB:.6f})"
                )
                sys.stdout.flush()

                if i % (iv1 * 100) == 0:
                    print("")
                    age_of_best = len(nmse_history) - nmse_history.index(min(nmse_history)) - 1
                    if age_of_best * iv1 > better_wait:
                        print(f"早停: 最佳 NMSE 已 {age_of_best * iv1} 迭代未更新")
                        break

            # 训练一步
            rand_index = np.random.choice(trainingsize, size=64, replace=False)
            h = ht[..., rand_index]
            h = np.transpose(h, [2, 0, 1])
            sess.run(train_, feed_dict={h_: h, A: A1})

        # 保存最优参数到 prev_layer_vars 以供下一层继承
        prev_layer_vars = best_var_values

        # 记录训练结果
        done.append(name)
        result_log = f"{name} NMSE={10 * np.log10(best_nmse):.6f} dB 在 {i} 次迭代"
        log.append(result_log)
        layernmse.append(10 * np.log10(best_nmse))

        # 保存最优参数到 state
        best_vars[name] = best_var_values
        state['done'] = done
        state['log'] = log
        state['layernmse'] = layernmse
        state['best_vars'] = best_vars

        try:
            save_trainable_vars(sess, savefile, snr=snr, **state)
            print(f"已保存训练状态到 {savefile}")
        except Exception as e:
            print(f"警告: 保存训练状态失败，错误: {e}")

    print(f"\n每层 NMSE (dB): {layernmse}")
    return sess







