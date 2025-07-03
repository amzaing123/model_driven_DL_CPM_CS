import tensorflow as tf
import numpy as np
import myshrinkage
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

###函数通过多层迭代和非线性收缩操作构建了一个自定义的 LAMP 网络，用于信号重建问题。每一层都进行了初始化和参数更新，最终返回构建的网络层次结构和占位符。
def build_LAMP(M, N, K,  snr, T, shrink, untied):#自定义LAMP网络，训练网络的每一层和后续网络的每一层
    eta, theta_init = myshrinkage2.get_shrinkage_function(shrink)
    layer = []
    var_all = []
    A_ = tf.placeholder(tf.complex128, (M, N))  # 输入A ###注意输入格式
    # 重塑 A_ 为 [None*M, N]
    # A_ = tf.reshape(A_, (-1, N))  # [None*M, N]

    h_ = tf.placeholder(tf.complex128, (None, N, K))  # 输入H ###注意输入格式
    h1 = tf.transpose(h_, [1, 0, 2])                        # [N, None, K]
    h1 = tf.reshape(h1, (N, -1))                     #[N, None*K]

    with tf.device('/cpu:0'):  # 将矩阵乘法放在 CPU 上
        ytemp1 = tf.matmul(A_, h1) #ytemp1 的形状应为 [M, None*K] = [32, 2]
        noise=gen_noise(ytemp1,snr)
        ytemp1=ytemp1+noise
    ytemp1 = tf.reshape(ytemp1, (M, tf.shape(h_)[0], K)) #(M,NONE,K)
    ytemp_ = tf.transpose(ytemp1, [1, 0, 2])                  #(NONE,M,K)
    y_ = ytemp_
    # first layer 初始化v0=0，h0=0，故v1=y
    v_0 = y_  # 残差为y   #(NONE,M,K)  #此时v_为v0

    OneOverMK = tf.constant(float(1) / (M*K), dtype=tf.float64)
    # bgb =  tf.expand_dims(tf.square(tf.norm(tf.abs(v_0), axis=[1, 2])), 1)
    # print("bgb dtype:", bgb.dtype, "bgb shape:", bgb.shape)
    rvar_1 = OneOverMK * tf.expand_dims(tf.square(tf.norm(tf.abs(v_0), axis=[1, 2])), 1)
    ### 61
    # U_ = tf.placeholder(tf.complex128, (G, N))
    A_H = tf.transpose(A_, conjugate=True)  ##共轭转置  # [N,M]
    with tf.device('/cpu:0'):  # 将矩阵乘法放在 CPU 上
        # B0 =A_H / (1.01 * tf.linalg.norm(A_H, ord=2))  # 初始化B [N,M]  B0
        B0 =A_H  # 初始化B [N,M]  B0
    Breal_ = tf.Variable(tf.real(B0), name='Breal_' + str(snr) + '_1')
    var_all.append(Breal_)
    Bimag_ = tf.Variable(tf.imag(B0), name='Bimag_' + str(snr) + '_1')
    var_all.append(Bimag_)
    B_ = tf.complex(Breal_, Bimag_, name='B') #(N, M)
    v0 = tf.transpose(v_0, [1, 0, 2])  #(M,NONE,K)
    v0 = tf.reshape(v0, (M, -1)) #(M,None*K)    [N, None, K]tf.reshape(h1, (N, -1)) 变为 [N, None*K]

    with tf.device('/cpu:0'):  # 将矩阵乘法放在 CPU 上
        Bvtemp0 = tf.matmul(B_, v0) # (N,M) * (M,None*K) =(N,None*K)
    Bv0 = tf.reshape(Bvtemp0, (M, tf.shape(h_)[0], K))  ###G存疑 (M,None,K)
    Bv_1 = tf.transpose(Bv0, [1, 0, 2])   ## (None,M,K)  X0初始值为0，Rt即为Bv_

    # x_gain = tf.reduce_sum(tf.square(tf.abs(Bv_)), axis=2)
    theta_ = tf.Variable(theta_init, dtype=tf.float64, name='theta_' + str(snr) + '_1' )
    # theta_ = tf.expand_dims(theta_, 0)
    var_all.append(theta_)  #theta0

    # print("Bv_1 dtype:", Bv_1.dtype, "shape:", Bv_1.shape)
    # print("rvar_1 dtype:", rvar_1.dtype, "shape:", rvar_1.shape)
    # print("Bv_1 dtype:", Bv_1.dtype, "shape:", Bv_1.shape)
    # print("rvar_1 dtype:", rvar_1.dtype, "shape:", rvar_1.shape)
    # print("theta_ dtype:", theta_.dtype, "shape:", theta_.shape)
    # xhat_, dxdr_1 = eta(Bv_1, rvar_1, K, theta_)  # xhat_ (None, M, K) xhat_1   此为bg是收缩函数 rvar_1,float64 rvar_1(None,K)
    xhat_, dxdr_1 = eta(Bv_1, rvar_1, theta_)  # xhat_ (None, M, K) xhat_1   此为bg是收缩函数 rvar_1,float64 rvar_1(None,K)

    #dxdr_1,complex128
    # xhat_, dxdr_1 = eta(Bv_1, rvar_1,  theta_)  # xhat_ (None, M, K) xhat_1   此为soft是收缩函数
    GOverM = tf.constant(float(M) / M, dtype=tf.complex128)
    b_1 = tf.expand_dims(GOverM * dxdr_1, 1)
    print("dxdr_1 dtype:", dxdr_1.dtype, "shape:", dxdr_1.shape)
    vb = tf.reshape(v0, [tf.shape(h_)[0], M * K])  # (None, M*K)
    bv = tf.multiply(b_1, vb)  # (None,M*K) b1*v0
    bv_ = tf.reshape(bv, [tf.shape(h_)[0], M, K])  # (None, M,K)

    x2 = tf.transpose(xhat_, [1, 0, 2])  # (M,None,K)
    x3 = tf.reshape(x2, (M, -1))  # (M,None*K)
    with tf.device('/cpu:0'):  # 将矩阵乘法放在 CPU 上
        Axhat1 = tf.matmul(A_, x3)  # (M,N) *(M,None*K)
    Axhat1 = tf.reshape(Axhat1, (M, tf.shape(h_)[0], K))  # (M,None,K)
    Axhat_1 = tf.transpose(Axhat1, [1, 0, 2])  # (None, M,K)
    v_ = y_ - Axhat_1 + bv_  # (None, M,K)  #2025-5-27根据算法发现循环中残差v使用的为初始值，理论应该为v1，添加
    # NOverM = tf.constant(float(N) / M, dtype=tf.complex128)
    # layer.append(('LAMP-{0} linear T=1'.format(shrink), Bv_, (Breal_, Bimag_), tuple(var_all), (0,)))
    # layer.append(('LAMP-{0} non-linear T=1'.format(shrink), xhat_, (theta_,), tuple(var_all), (1,)))
    layer.append(('LAMP-{0} T=1'.format(shrink), xhat_, tuple(var_all), tuple(var_all), (0,)))
    ###第一层输出（x1，v1）（xhat_，v_）
    for t in range(2, T+1): ###从2到T，2，3，4，5
        if untied:  # 表明每一层的B都会训练
            Breal_ = tf.Variable(tf.real(B0), name='Breal_' + str(snr) + '_' + str(t))
            var_all.append(Breal_)
            Bimag_ = tf.Variable(tf.imag(B0), name='Bimag_' + str(snr) + '_' + str(t))
            var_all.append(Bimag_)
            B_ = tf.complex(Breal_, Bimag_, name='B')  #(N,M)
            v3 = tf.transpose(v_, [1, 0, 2]) #(M,None, K)
            v4 = tf.reshape(v3, (M, -1)) #(M,None*K)
            with tf.device('/cpu:0'):  # 将矩阵乘法放在 CPU 上
                Bv_temp = tf.matmul(B_, v4)#(N,M) *(M,None*K)= (N,None*K)
            Bv = tf.reshape(Bv_temp, (M, tf.shape(h_)[0], K))
            Bv_ = tf.transpose(Bv, [1, 0, 2])#(None, M, K)
            rhat_ = xhat_ + Bv_ #(None,M,K) ##(None, M, K)
        else:
            v3 = tf.transpose(v_, [1, 0, 2])
            v4 = tf.reshape(v3, (M, -1))
            with tf.device('/cpu:0'):  # 将矩阵乘法放在 CPU 上
                Bv_temp = tf.matmul(B_, v4)
            Bv = tf.reshape(Bv_temp, (M, tf.shape(h_)[0], K))
            Bv_ = tf.transpose(Bv, [1, 0, 2])
            rhat_ = xhat_ + Bv_
        theta_ = tf.Variable(theta_init, dtype=tf.float64, name='theta_' + str(snr) + '_' + str(t))
        var_all.append(theta_)
        rvar_ = OneOverMK * tf.expand_dims(tf.square(tf.norm(tf.abs(v_), axis=[1, 2])), 1)
        # xhat_, dxdr_ = eta(rhat_, rvar_, K, theta_)  # xhat_ (None,M,K) bg
        xhat_, dxdr_ = eta(rhat_, rvar_, theta_)  # xhat_ (None,M,K)  soft
        b_ = tf.expand_dims(GOverM * dxdr_, 1)
        # print("b_ dtype:", b_.dtype, "shape:", b_.shape)
        v_1 = tf.reshape(v_, [tf.shape(h_)[0], M * K])  # (None, M*K)
        # v_1 =tf.reshape(v_1, [tf.shape(h_)[0], M, K])  # (None, M,K)
        bv = tf.multiply(b_, v_1)  # (None, M,K)
        bv_ = tf.reshape(bv, [tf.shape(h_)[0], M, K])  # (None, M,K)
        with tf.device('/cpu:0'):  # 将矩阵乘法放在 CPU 上
            matrix =A_  #(M,N)
        x2 = tf.transpose(xhat_, [1, 0, 2]) #  (M,None,K)
        x3 = tf.reshape(x2, (M, -1))  #  (M,None*K)
        with tf.device('/cpu:0'):  # 将矩阵乘法放在 CPU 上
            Axhat = tf.matmul(matrix, x3)# (M,N) *(M,None*K)
        Axhat = tf.reshape(Axhat, (M, tf.shape(h_)[0], K))#(M,None,K)
        Axhat_ = tf.transpose(Axhat, [1, 0, 2]) #(None, M,K)
        v_ = y_ - Axhat_ + bv_ #(None, M,K)
        # layer.append(('LAMP-{0} linear T={1}'.format(shrink, t), rhat_, (Breal_, Bimag_), tuple(var_all), (0,)))
        # layer.append(('LAMP-{0} non-linear T={1}'.format(shrink, t), xhat_, (theta_,), tuple(var_all), (1,)))
        layer.append(('LAMP-{0} T={1}'.format(shrink, t), xhat_, tuple(var_all), tuple(var_all), (0,)))
    return layer, h_, A_
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


            # # 在 nmse_ 计算部分添加数值稳定性保护
            # epsilon = 1e-10
            # nmse_ = tf.reduce_mean(
            #     tf.square(tf.norm(tf.abs(hhat_ - x_), axis=[1, 2])) /
            #     tf.maximum(tf.square(tf.norm(tf.abs(x_), axis=[1, 2])), epsilon)
            # )

            # 添加调试日志以跟踪 NMSE 和信号范数
            tf.print("Layer:", name, "NMSE:", nmse_, "Signal Norm:", tf.norm(tf.abs(x_), axis=[1, 2]),
                     output_stream=sys.stdout)

            loss_ = nmse_

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
        tv = dict([(str(v.name).replace(':', '_'), v) for v in variables])
        for k, d in loadmat(filename).items():  # (k, d)表示字典中的(键，值)
            if k in tv:
                print('restore ' + k)
                sess.run(tf.assign(tv[k], d))
                # print(sess.run(tv[k]))
            else:
                if k == 'done':
                    for i in range(0, len(d)):
                        a = d[i]
                        d[i] = a.strip()
                other[k] = d
                # print('error!')
    except IOError:
        pass
    return other

def save_trainable_vars(sess, filename, snr, **kwargs):
    save = {}
    for v in tf.trainable_variables():
        if str(v.name).split('_')[1] == str(snr):
            save[str(v.name).replace(':', '_')] = sess.run(v)
        continue
        # save[str(v.name)] = sess.run(v)
    save.update(kwargs)
    savemat(filename, save)
    # np.savez(filename, **save)

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

def do_training(h_, training_stages, A, savefile, trainingfile, validationfile, snr, iv1=100, maxit=50000, better_wait=5000):
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

    # 初始化 TensorFlow 会话
    sess = tf.Session()
    sess.run(tf.global_variables_initializer(), feed_dict={A: A1})

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


