import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import myshrinkage
from scipy.io import loadmat, savemat
def gen_noise(y, snr):  # 定义一个函数，用于生成噪声
    ypower = tf.reduce_mean(tf.square(tf.abs(y)), axis=0, keepdims=True)  # 计算信号 y 的平均功率
    noise_var = tf.cast(ypower / ((10 ** (snr / 10)) * 2), tf.complex128)  # 计算噪声方差，转换为复数类型
    noise = tf.complex(real=tf.random_normal(shape=tf.shape(y), dtype=tf.float64),  # 生成服从标准正态分布的噪声，实部
                       imag=tf.random_normal(shape=tf.shape(y), dtype=tf.float64))  # 虚部
    n = tf.sqrt(noise_var) * noise  # 计算噪声

    return n  # 返回噪声

def build_LAMP(M, N, K, snr, T, shrink, untied, Breal_vals=None, Bimag_vals=None, theta_vals=None):
    """
    构建 LAMP 网络，每层包含线性变换和非线性收缩操作。

    参数：
        M, N, K: 网络维度参数
        snr: 信噪比 (dB)
        T: 网络层数
        shrink: 收缩函数类型（如 'bg'）
        untied: 是否为每层训练独立的 B 矩阵
        Breal_vals: 预训练的 Breal 权重列表，形状 [T, N, M]
        Bimag_vals: 预训练的 Bimag 权重列表，形状 [T, N, M]
        theta_vals: 预训练的 theta 参数列表，形状 [T, ...]

    返回：
        layer: 网络层列表，每层包含 (名称, 输出, 变量, 所有变量, 元组)
        h_: 输入信号占位符
        A_: 输入矩阵占位符
    """
    # 获取收缩函数及其初始 theta 值
    eta, theta_init = myshrinkage.get_shrinkage_function(shrink)
    layer = []
    var_all = []

    # 定义输入占位符
    A_ = tf.placeholder(tf.complex128, (M, N), name='A')  # 输入矩阵 A
    h_ = tf.placeholder(tf.complex128, (None, N, K), name='h')  # 输入信号 h
    h1 = tf.transpose(h_, [1, 0, 2])  # [N, None, K]
    h1 = tf.reshape(h1, (N, -1))  # [N, None*K]

    # 计算初始输出 y
    with tf.device('/cpu:0'):  # 矩阵乘法在 CPU 上执行
        ytemp1 = tf.matmul(A_, h1)  # [M, None*K]
        noise = gen_noise(ytemp1, snr)  # 可选：添加噪声
        ytemp1 = ytemp1 + noise
    ytemp1 = tf.reshape(ytemp1, (M, tf.shape(h_)[0], K))  # [M, None, K]
    y_ = tf.transpose(ytemp1, [1, 0, 2])  # [None, M, K]
    v_ = y_  # 初始残差 v = y

    # 计算初始残差方差
    OneOverMK = tf.constant(float(1) / (M * K), dtype=tf.float64)
    rvar_ = OneOverMK * tf.expand_dims(tf.square(tf.norm(tf.abs(v_), axis=[1, 2])), 1)

    # 初始化 B 矩阵（A 的共轭转置）
    A_H = tf.transpose(A_, conjugate=True)  # [N, M]
    with tf.device('/cpu:0'):
        B = A_H  # 初始化 B [N, M]

    # 第一层
    if Breal_vals is not None and Bimag_vals is not None and theta_vals is not None:
        Breal_ = tf.Variable(Breal_vals[0], name=f'Breal_{snr}_1', dtype=tf.float64)
        Bimag_ = tf.Variable(Bimag_vals[0], name=f'Bimag_{snr}_1', dtype=tf.float64)
        theta_ = tf.Variable(theta_vals[0], name=f'theta_{snr}_1', dtype=tf.float64)
    # else:
        # Breal_ = tf.Variable(tf.real(B), name=f'Breal_{snr}_1', dtype=tf.float64)
        # Bimag_ = tf.Variable(tf.imag(B), name=f'Bimag_{snr}_1', dtype=tf.float64)
        # theta_ = tf.Variable(theta_init, name=f'theta_{snr}_1', dtype=tf.float64)

    var_all.extend([Breal_, Bimag_, theta_])
    B_ = tf.complex(Breal_, Bimag_, name='B')  # [N, M]

    # 第一层线性变换
    v1 = tf.transpose(v_, [1, 0, 2])  # [M, None, K]
    v1 = tf.reshape(v1, (M, -1))  # [M, None*K]
    with tf.device('/cpu:0'):
        Bvtemp = tf.matmul(B_, v1)  # [N, None*K]
    Bv = tf.reshape(Bvtemp, (M, tf.shape(h_)[0], K))  # [M, None, K]
    Bv_ = tf.transpose(Bv, [1, 0, 2])  # [None, M, K]

    # 第一层非线性收缩
    xhat_, dxdr_ = eta(Bv_, rvar_, K, theta_)  # xhat_ [None, M, K]

    # 计算 Onsager 修正项
    GOverM = tf.constant(float(M) / M, dtype=tf.complex128)
    b_ = tf.expand_dims(GOverM * dxdr_, 1)
    v_1 = tf.reshape(v_, [tf.shape(h_)[0], M * K])  # [None, M*K]
    bv = tf.multiply(b_, v_1)  # [None, M*K]
    bv_ = tf.reshape(bv, [tf.shape(h_)[0], M, K])  # [None, M, K]

    # 更新残差
    x2 = tf.transpose(xhat_, [1, 0, 2])  # [M, None, K]
    x3 = tf.reshape(x2, (M, -1))  # [M, None*K]
    with tf.device('/cpu:0'):
        Axhat = tf.matmul(A_, x3)  # [M, None*K]
    Axhat = tf.reshape(Axhat, (M, tf.shape(h_)[0], K))  # [M, None, K]
    Axhat_ = tf.transpose(Axhat, [1, 0, 2])  # [None, M, K]
    v_ = y_ - Axhat_ + bv_  # [None, M, K]

    # 记录第一层
    layer.append((f'LAMP-{shrink} T=1', xhat_, tuple(var_all), tuple(var_all), (0,)))

    # 后续层 (t=2 到 T)
    for t in range(2, T + 1):
        # 计算残差方差
        rvar_ = OneOverMK * tf.expand_dims(tf.square(tf.norm(tf.abs(v_), axis=[1, 2])), 1)

        # 根据 untied 参数决定是否使用独立的 B 矩阵
        if untied:
            if Breal_vals is not None and Bimag_vals is not None:
                Breal_ = tf.Variable(Breal_vals[t - 1], name=f'Breal_{snr}_{t}', dtype=tf.float64)
                Bimag_ = tf.Variable(Bimag_vals[t - 1], name=f'Bimag_{snr}_{t}', dtype=tf.float64)
            else:
                Breal_ = tf.Variable(tf.real(B), name=f'Breal_{snr}_{t}', dtype=tf.float64)
                Bimag_ = tf.Variable(tf.imag(B), name=f'Bimag_{snr}_{t}', dtype=tf.float64)
            var_all.extend([Breal_, Bimag_])
            B_ = tf.complex(Breal_, Bimag_, name='B')  # [N, M]
            v3 = tf.transpose(v_, [1, 0, 2])  # [M, None, K]
            v4 = tf.reshape(v3, (M, -1))  # [M, None*K]
            with tf.device('/cpu:0'):
                Bv_temp = tf.matmul(B_, v4)  # [N, None*K]
            Bv = tf.reshape(Bv_temp, (M, tf.shape(h_)[0], K))  # [M, None, K]
            Bv_ = tf.transpose(Bv, [1, 0, 2])  # [None, M, K]
            rhat_ = xhat_ + Bv_  # [None, M, K]
        else:
            v3 = tf.transpose(v_, [1, 0, 2])
            v4 = tf.reshape(v3, (M, -1))
            with tf.device('/cpu:0'):
                Bv = tf.matmul(B_, v4)
            Bv = tf.reshape(Bv, (M, tf.shape(h_)[0], K))
            Bv_ = tf.transpose(Bv, [1, 0, 2])
            rhat_ = xhat_ + Bv_

        # 非线性收缩
        if theta_vals is not None:
            # theta_ = tf.Variable(theta_vals[t - 1], name=f'theta_{snr}_{t}', dtype=tf.float64)
            theta_ = tf.Variable(theta_vals[t - 1], name=f'theta_{snr}_1_0_0', dtype=tf.float64)
        # else:
        #     theta_ = tf.Variable(theta_init, name=f'theta_{snr}_{t}', dtype=tf.float64)
        var_all.append(theta_)
        xhat_, dxdr_ = eta(rhat_, rvar_, K, theta_)  # [None, M, K]

        # 更新 Onsager 修正和残差
        b_ = tf.expand_dims(GOverM * dxdr_, 1)
        v_1 = tf.reshape(v_, [tf.shape(h_)[0], M * K])  # [None, M*K]
        bv = tf.multiply(b_, v_1)  # [None, M*K]
        bv_ = tf.reshape(bv, [tf.shape(h_)[0], M, K])  # [None, M, K]
        x2 = tf.transpose(xhat_, [1, 0, 2])  # [M, None, K]
        x3 = tf.reshape(x2, (M, -1))  # [M, None*K]
        with tf.device('/cpu:0'):
            Axhat = tf.matmul(A_, x3)  # [M, None*K]
        Axhat = tf.reshape(Axhat, (M, tf.shape(h_)[0], K))  # [M, None, K]
        Axhat_ = tf.transpose(Axhat, [1, 0, 2])  # [None, M, K]
        v_ = y_ - Axhat_ + bv_  # [None, M, K]

        # 记录当前层
        layer.append((f'LAMP-{shrink} T={t}', xhat_, tuple(var_all), tuple(var_all), (0,)))

    return layer, h_, A_

def test_LAMP(M, N, K, snr, T, shrink, untied, mat_file_path, A_test, h_test):
    """
    测试 LAMP 网络，评估恢复信号的性能并可视化结果。

    参数：
        M, N, K: 网络维度参数
        snr: 信噪比 (dB)
        T: 网络层数
        shrink: 收缩函数类型（如 'bg'）
        untied: 是否为每层训练独立的 B 矩阵
        mat_file_path: 预训练权重 .mat 文件路径
        A_test: 测试输入矩阵 A，形状 [M, N]
        h_test: 测试输入信号 h，形状 [N, K, trainingsize]

    返回：
        无（打印 NMSE 并显示可视化图）
    """
    # 加载 .mat 文件中的预训练权重
    try:
        mat_data = sio.loadmat(mat_file_path)
    except Exception as e:
        raise ValueError(f"加载 .mat 文件失败: {e}")

    # 提取 Breal_, Bimag_, theta_ 的值
    Breal_vals, Bimag_vals, theta_vals = [], [], []
    for t in range(1, T + 1):
        # 与 build_LAMP 的命名规则保持一致
        Breal_key = f'Breal_{snr}_{t}_0'
        Bimag_key = f'Bimag_{snr}_{t}_0'
        theta_key = f'theta_{snr}_{t}_0'  # 更新为与 build_LAMP 一致的命名

        try:
            Breal_vals.append(mat_data[Breal_key])  # 预期形状: [N, M]
            Bimag_vals.append(mat_data[Bimag_key])  # 预期形状: [N, M]
            theta_vals.append(mat_data[theta_key].flatten())  # 展平为一维数组
        except KeyError as e:
            raise KeyError(f".mat 文件中缺少键: {e}")

    # 重建 LAMP 网络，使用加载的权重
    layer, h_, A_ = build_LAMP(M, N, K, snr, T, shrink, untied, Breal_vals, Bimag_vals, theta_vals)

    # 初始化 TensorFlow 会话
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # 准备测试数据
        trainingsize = np.size(h_test, axis=2)
        rand_index = np.random.choice(trainingsize, size=64, replace=False)
        h = h_test[..., rand_index]  # 随机选择 64 个样本
        h_test_input = np.transpose(h, [2, 0, 1])  # 形状: [64, N, K]

        # 运行最后一层输出
        xhat_ = sess.run(layer[-1][1], feed_dict={A_: A_test, h_: h_test_input})

        # 计算 NMSE
        mse = np.mean(np.abs(xhat_ - h_test_input) ** 2)
        print(f"MSE:{mse:.6f}")
        nmse = mse / np.mean(np.abs(h_test_input) ** 2)
        nmse_db=10 * np.log10(nmse)
        print(f"NMSE: {nmse:.6f}")
        print(f"NMSEdb: {nmse_db:.6f}")

        # 设置 matplotlib 中文字体
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
        plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

        # 可视化信号对比（实部和虚部）
        for k in range(min(K, 2)):  # 最多显示 2 个用户
            plt.figure(figsize=(12, 5))

            # 实部子图
            plt.subplot(1, 2, 1)
            plt.plot(np.real(h_test_input[0, :, k]), 'b-', label='真实信号 (实部)')
            plt.plot(np.real(xhat_[0, :, k]), 'r--', label='恢复信号 (实部)')
            plt.title(f'用户 {k + 1} - 实部')
            plt.xlabel('样本索引')
            plt.ylabel('幅度')
            plt.legend()
            plt.grid(True)

            # 虚部子图
            plt.subplot(1, 2, 2)
            plt.plot(np.imag(h_test_input[0, :, k]), 'g-', label='真实信号 (虚部)')
            plt.plot(np.imag(xhat_[0, :, k]), 'y--', label='恢复信号 (虚部)')
            plt.title(f'用户 {k + 1} - 虚部')
            plt.xlabel('样本索引')
            plt.ylabel('幅度')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.show()

        # # 可视化信号对比（实部和虚部 - 点图版）
        # for k in range(min(K, 2)):  # 最多显示 2 个用户
        #     plt.figure(figsize=(12, 5))
        #
        #     # 实部子图 - 点图显示
        #     plt.subplot(1, 2, 1)
        #     plt.scatter(range(len(h_test_input[0, :, k])), np.real(h_test_input[0, :, k]),
        #                 c='blue', marker='o', s=30, alpha=0.7, label='真实信号 (实部)')
        #     plt.scatter(range(len(xhat_[0, :, k])), np.real(xhat_[0, :, k]),
        #                 c='red', marker='x', s=30, alpha=0.7, label='恢复信号 (实部)')
        #     plt.title(f'用户 {k + 1} - 实部')
        #     plt.xlabel('样本索引')
        #     plt.ylabel('幅度')
        #     plt.legend()
        #     plt.grid(True)
        #
        #     # 虚部子图 - 点图显示
        #     plt.subplot(1, 2, 2)
        #     plt.scatter(range(len(h_test_input[0, :, k])), np.imag(h_test_input[0, :, k]),
        #                 c='green', marker='o', s=30, alpha=0.7, label='真实信号 (虚部)')
        #     plt.scatter(range(len(xhat_[0, :, k])), np.imag(xhat_[0, :, k]),
        #                 c='red', marker='x', s=30, alpha=0.7, label='恢复信号 (虚部)')
        #     plt.title(f'用户 {k + 1} - 虚部')
        #     plt.xlabel('样本索引')
        #     plt.ylabel('幅度')
        #     plt.legend()
        #     plt.grid(True)
        #
        #     plt.tight_layout()
        #     plt.show()


# 示例调用
M, N, K, snr, T, shrink, untied = 16, 16, 1, 17, 5, 'bg', True
mat_file_path = "cpm_bg_16_16_train1carriers17dB_0605_2.mat"  # .mat 文件路径
# mat_file_path = "test_30db_0603.mat"  # .mat 文件路径
# testfile = "test.mat"
testfile = "test1.mat"
Dtest = loadmat(testfile)
A_test = Dtest['st02']  # 测试矩阵 A
# h_test = Dtest['h_mc_test']  # 测试信号 h
h_test = Dtest['h_mc_test']  # 测试信号 h
test_LAMP(M, N, K, snr, T, shrink, untied, mat_file_path, A_test, h_test)