# import LAMP_CE_Network
import CPM_LAMP_CE_Network1
import os
import tensorflow as tf

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # BE QUIET!!!!  设置log日志级别，只显示warning和error
if tf.test.is_gpu_available():
    print("GPU 设备可用。")
config = tf.compat.v1.ConfigProto()
# 设置最大占有GPU不超过显存的80%（可选）
# config.gpu_options.per_process_gpu_memory_fraction=0.8
config.gpu_options.allow_growth = True  # 设置动态分配GPU内存
sess = tf.compat.v1.Session(config=config)

type = 'cpm'  # 'UPA'
shrink = 'pwgrid'  # 'soft' 'pwgrid'
M_N = '16_16'
M = 16   ###原为
N = 16
# G = 16 ###原为1024
K = 1
# L = 8
T = 5
# SNRrange = [0, 5, 10, 15, 20]
SNRrange = [0]


for snr in SNRrange:
    # savenetworkfilename = type + '_' + shrink + '_' + str(snr) + 'dB.mat'
    savenetworkfilename = type + '_' + shrink + '_' + M_N + '_train' + str(K) + 'carriers' + str(snr) + 'dB.mat'
    # trainingfilename = type + 'traindata' + str(N) + '_' + str(K) + '.mat'
    trainingfilename = 'train1.mat'
    # validationfilename = type + 'validationdata' + str(N) + '_' + str(K) + '.mat'
    validationfilename = 'validation1.mat'

    layers, h_, A = CPM_LAMP_CE_Network1.build_LAMP(M=M, N=N,  K=K,  snr=snr, T=T, shrink=shrink, untied=False)
    training_stages = CPM_LAMP_CE_Network1.setup_training(layers, h_, A,  K, N, M, trinit=1e-3, refinements=(0.5,))
    sess = CPM_LAMP_CE_Network1.do_training(h_=h_, training_stages=training_stages, A=A, savefile=savenetworkfilename,
                                       trainingfile=trainingfilename, validationfile=validationfilename, maxit=1000000,
                                       snr=snr)
