import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import myshrinkage1
from scipy.io import loadmat, savemat
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

def build_LVAMP_dense(M, N, K,  snr, T, shrink):
    """ Builds the non-SVD (i.e. dense) parameterization of LVAMP
    and returns a list of trainable points(name,xhat_,newvars)
    """
    eta,theta_init = myshrinkage1.get_shrinkage_function(shrink)
    layers=[]
    var_all =[]
    A_ = tf.placeholder(tf.complex128, (M, N), name='A')  # 输入A ###注意输入格式
    h_ = tf.placeholder(tf.complex128, (None, N, K), name='h')  # 输入H ###注意输入格式
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

    vs_def = tf.constant(1, dtype=tf.float64)  # 可能存在问题 #应该为复数，暂不确定

    theta_ = tf.Variable(theta_init, dtype=tf.float64, name='theta_' + str(snr) + '_1')
    vs_ = tf.Variable(vs_def, name='vs_'+ str(snr) + '_1')
    var_all.append(theta_)
    var_all.append(vs_)

    rhat_ = xhat_init#(NONE,N,K)
    with tf.device('/cpu:0'):
        y_1=(tf.expand_dims(tf.square(tf.norm(tf.abs(y_), axis=[1, 2])), 1))/N

        rvar_=vs_*y_1 #float64 (none,k)

    xhat_,dxdr_ = eta(rhat_,rvar_,K,theta_ )  #xhat_nl_,complex128  alpha_nl_,complex128 # xhat_ (None, M, K)

    layers.append( ('LVAMP-{0} T={1}'.format(shrink,1),xhat_, tuple(var_all), tuple(var_all), (0,)) )


    for t in range(2,T+1):
        ###  rhat 网络上边
        dxdr_ = tf.reduce_mean(dxdr_, axis=0)  # each col average dxdr
        one = tf.complex(tf.constant(1.0, dtype=tf.float64), tf.constant(0.0, dtype=tf.float64))
        with tf.device('/cpu:0'):
            gain_r = tf.divide(one, one - dxdr_) # dxdr_ ()
        dxdr_r = dxdr_ * rhat_
        rhat_ = gain_r * (xhat_ - dxdr_r)

        ###rvar 网络下边
        dxdr_ = tf.math.abs(dxdr_)
        with tf.device('/cpu:0'):
            gain_v1 = tf.divide(1, 1 - dxdr_)
        gain_v2 = gain_v1 * dxdr_
        gain_v = tf.sqrt(gain_v2)
        rvar_ = rvar_ * gain_v



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

        with (tf.device('/cpu:0')):
            xhat_1 = tf.matmul(H_,y_2)   # H (N,M) * y_2(M,NONE*K) xhat_1 (N,NONE*K)

            xhat_2 = tf.matmul(G_,rhat_)#(N,M) * rhat_(M,NONE*K) xhat_2 (N,NONE*K)

            xhat_ =xhat_1+xhat_2  # xhat_(N,NONE*K)

        xhat_=tf.reshape(xhat_, (tf.shape(h_)[0],M,K))

        dxdr_ = tf.expand_dims(tf.diag_part(G_),1)

        # 对复数张量 dxdr_ 求模
        real_part = tf.cast(.5 / N, dtype=tf.float64)
        imag_part = tf.cast(0, dtype=tf.float64)
        eps = tf.complex(real_part, imag_part)

        dxdr_=clip_complex_dxdr(dxdr_,eps) #比较大小确定阈值

        vs_= tf.Variable(vs_def, name='vs_'+ str(snr)+ '_'+str(t), dtype=tf.float64)
        var_all.append(vs_)
        ###rhat 网络上层
        dxdr_ = tf.reduce_mean(dxdr_, axis=0)  # each col average dxdr  #新添
        one1 = tf.complex(tf.constant(1.0, dtype=tf.float64), tf.constant(0.0, dtype=tf.float64))


        with tf.device('/cpu:0'):
            gain_r = tf.divide(one1, one1 - dxdr_) # 1-vt  dxdr_(1,)

        dxdr_r1 = dxdr_ * rhat_  # dxdr_r(16, ?)  rhat_ shape: (16, ?)
        dxdr_r2=tf.reshape(dxdr_r1,(M, tf.shape(h_)[0], K)) #(M,None,K)
        dxdr_r=tf.transpose(dxdr_r2, [1, 0, 2])#(None,M,K)

        rhat_ = gain_r * (xhat_ - dxdr_r) #xhat_(None,M,K)   (?, 16, ?)现在不对
        #rvar 网络下层
        dxdr_ = tf.math.abs(dxdr_)
        with tf.device('/cpu:0'):
            gain_v1 = tf.divide(1, 1 - dxdr_)
        gain_v2 = gain_v1 * dxdr_
        gain_v = tf.sqrt(gain_v2)
        gain_vs=gain_v*vs_
        rvar_ = rvar_ * gain_vs         #现在存在报错，维度不匹配

        rhat_=tf.reshape(rhat_,(M, tf.shape(h_)[0], K))
        rhat_ = tf.transpose(rhat_, [1, 0, 2])  # (NONE,M,K)
        theta_ = tf.Variable(theta_init,name='theta_'+ str(snr)+ '_'+str(t),dtype=tf.float64)
        var_all.append(theta_)

        xhat_,dxdr_ = eta(rhat_,rvar_,K,theta_ )#rvar_ float64   rhat_ complex128 (None,M,K)

        layers.append( ('LVAMP-{0} T={1}'.format(shrink,t),xhat_, tuple(var_all), tuple(var_all), (0,)) )

    return layers ,h_, A_


def test_LVAMP(M, N, K, snr, T, shrink, mat_file_path, A_test, h_test):
    """
    测试 LVAMP 网络，评估恢复信号的性能并可视化结果。
    """
    # 调试：打印输入数据的形状和类型
    print(f"A_test shape: {A_test.shape}, dtype: {A_test.dtype}")
    print(f"h_test shape: {h_test.shape}, dtype: {h_test.dtype}")

    # 转换为复数类型并验证形状
    if A_test.dtype not in [np.complex128, np.complex64]:
        A_test = A_test.astype(np.complex128)
    if h_test.dtype not in [np.complex128, np.complex64]:
        h_test = h_test.astype(np.complex128)

    if A_test.shape != (M, N):
        raise ValueError(f"A_test shape {A_test.shape} does not match expected [{M}, {N}]")
    if h_test.shape[0] != N or h_test.shape[1] != K:
        raise ValueError(f"h_test shape {h_test.shape} does not match expected [N={N}, K={K}, trainingsize]")

    # 加载 .mat 文件中的预训练权重
    try:
        mat_data = sio.loadmat(mat_file_path)
    except Exception as e:
        raise ValueError(f"加载 .mat 文件失败: {e}")

    # 提取 Hreal_, Himag_, Greal_, Gimag_, vs_, theta_ 的值
    Hreal_vals, Himag_vals, Greal_vals, Gimag_vals, vs_vals, theta_vals = [], [], [], [], [], []
    for t in range(1, T + 1):
        Hreal_key = f'Hreal_{snr}_{t}_0'
        Himag_key = f'Himag_{snr}_{t}_0'
        vs_key = f'vs_{snr}_{t}_0'
        theta_key = f'theta_{snr}_{t}_0'

        try:
            Hreal_vals.append(mat_data[Hreal_key])  # [N, M]
            Himag_vals.append(mat_data[Himag_key])  # [N, M]
            vs_vals.append(mat_data[vs_key].flatten()[0])  # 确保为标量
            theta_vals.append(mat_data[theta_key].flatten())  # 展平
            if t >= 2:  # 从第 2 层开始加载 G 变量
                Greal_key = f'Greal_{snr}_{t}_0'
                Gimag_key = f'Gimag_{snr}_{t}_0'
                Greal_vals.append(mat_data[Greal_key])  # [N, N]
                Gimag_vals.append(mat_data[Gimag_key])  # [N, N]
            else:
                Greal_vals.append(None)  # 第一层无 G 变量，占位
                Gimag_vals.append(None)
        except KeyError as e:
            raise KeyError(f".mat 文件中缺少键: {e}")

    # 重建 LVAMP 网络，使用加载的权重
    layers, h_, A_ = build_LVAMP_dense(M, N, K, snr, T, shrink)
    print("Dependencies of layers[-1][1]:", layers[-1][1].op.inputs)

    # 初始化 TensorFlow 会话
    with tf.Session() as sess:
        # 提供临时的 feed_dict 给初始化
        init_feed_dict = {A_: np.zeros((M, N), dtype=np.complex128), h_: np.zeros((1, N, K), dtype=np.complex128)}
        sess.run(tf.global_variables_initializer(), feed_dict=init_feed_dict)

        var_map = {v.name: v for v in tf.global_variables()}

        # 2. 直接根据.mat文件中的键名来查找并分配权重
        print("--- Starting weight assignment ---")
        for t in range(1, T + 1):
            # 构建当前层的变量名称
            hreal_name = f'Hreal_{snr}_{t}:0'
            himag_name = f'Himag_{snr}_{t}:0'
            theta_name = f'theta_{snr}_{t}:0'
            vs_name = f'vs_{snr}_{t}:0'

            # 分配 H, theta, vs
            print(f"Assigning weights for Layer {t}")
            sess.run(tf.assign(var_map[hreal_name], Hreal_vals[t - 1]))
            sess.run(tf.assign(var_map[himag_name], Himag_vals[t - 1]))

            # 从.mat加载的theta需要reshape以匹配变量形状(2,1)
            theta_val_reshaped = np.array(theta_vals[t - 1]).reshape(-1, 1)
            sess.run(tf.assign(var_map[theta_name], theta_val_reshaped))

            sess.run(tf.assign(var_map[vs_name], float(vs_vals[t - 1])))

            # 从第二层开始，分配G
            if t >= 2:
                greal_name = f'Greal_{snr}_{t}:0'
                gimag_name = f'Gimag_{snr}_{t}:0'
                sess.run(tf.assign(var_map[greal_name], Greal_vals[t - 1]))
                sess.run(tf.assign(var_map[gimag_name], Gimag_vals[t - 1]))

        print("--- Weight assignment finished ---")

        # 准备测试数据
        trainingsize = np.size(h_test, axis=2)
        rand_index = np.random.choice(trainingsize, size=64, replace=False)
        h = h_test[..., rand_index]  # [16, 1, 64]
        h_test_input = np.transpose(h, [2, 0, 1])  # [64, 16, 1]
        print(f"h_test_input shape: {h_test_input.shape}, dtype: {h_test_input.dtype}")

        # 运行最后一层输出
        feed_dict = {A_: A_test, h_: h_test_input}
        print(f"feed_dict keys: {feed_dict.keys()}")
        print(f"feed_dict[A_] shape: {feed_dict[A_].shape}, dtype: {feed_dict[A_].dtype}")
        print(f"feed_dict[h_] shape: {feed_dict[h_].shape}, dtype: {feed_dict[h_].dtype}")
        try:
            xhat_ = sess.run(layers[-1][1], feed_dict=feed_dict)
        except tf.errors.InvalidArgumentError as e:
            print(f"Error details: {e}")
            raise

        # 计算 NMSE
        mse = np.mean(np.abs(xhat_ - h_test_input) ** 2)
        print(f"MSE: {mse:.6f}")
        nmse = mse / np.mean(np.abs(h_test_input) ** 2)
        nmse_db = 10 * np.log10(nmse)
        print(f"NMSE: {nmse:.6f}")
        print(f"NMSEdb: {nmse_db:.6f}")

        # 设置 matplotlib 中文字体
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
        plt.rcParams["axes.unicode_minus"] = False

        # 可视化信号对比
        for k in range(min(K, 2)):
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(np.real(h_test_input[0, :, k]), 'b-', label='真实信号 (实部)')
            plt.plot(np.real(xhat_[0, :, 0]), 'r--', label='恢复信号 (实部)')  # 修正索引为 0，因为 K=1
            plt.title(f'用户 {k + 1} - 实部')
            plt.xlabel('样本索引')
            plt.ylabel('幅度')
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(np.imag(h_test_input[0, :, k]), 'g-', label='真实信号 (虚部)')
            plt.plot(np.imag(xhat_[0, :, 0]), 'y--', label='恢复信号 (虚部)')  # 修正索引为 0
            plt.title(f'用户 {k + 1} - 虚部')
            plt.xlabel('样本索引')
            plt.ylabel('幅度')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.show()

# 示例调用
M, N, K, snr, T, shrink = 16, 16, 1, 17, 5, 'bg'
mat_file_path = "cpm_LVAMP_bg_16_16_1_train_layer_number5_17dB_0623_3.mat"
testfile = "test1.mat"
Dtest = loadmat(testfile)
A_test = Dtest['st02']
h_test = Dtest['h_mc_test']
test_LVAMP(M, N, K, snr, T, shrink, mat_file_path, A_test, h_test)

