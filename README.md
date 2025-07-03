# model_driven_DL_CPM_CS
# 基于CPM调制信号的信道估计算法研究

本项目致力于基于 CPM（Continuous Phase Modulation，连续相位调制）信号的信道估计算法研究，主要采用模型驱动的深度学习方法，包括 LAMP（Learned Approximate Message Passing，可学习近似消息传递）和 LVAMP（Learned Vector Approximate Message Passing，可学习向量近似消息传递）两种算法，实现对信道状态的高效估计。

## 项目背景

在无线通信系统中，信道估计是影响系统性能的关键技术之一。传统方法在复杂信道环境下性能有限。近年来，模型驱动深度学习方法结合了物理建模与数据驱动的优点，展现出优异的性能和泛化能力。

## 主要内容

- **CPM信号建模**：模拟连续相位调制信号在不同信道条件下的传输过程。
- **LAMP算法实现**：基于深度学习实现的可学习近似消息传递信道估计算法。
- **LVAMP算法实现**：基于深度学习实现的可学习向量近似消息传递信道估计算法。
- **性能对比与分析**：对比传统信道估计算法和模型驱动深度学习算法在CPM信号下的性能表现。

## 项目结构
- **主要程序**：CPM_LAMP_Network.py,CPM_LVAMP_Network.py,CPM_trainLAMP_CE.py,CPM_trainLVAMP_CE.py
## 环境依赖

- Python 3.7+
- TensorFlow1.13.1
- NumPy
- Matplotlib

## 参考文献
- Borgerding M, Schniter P. "AMP-Inspired Deep Networks for Sparse Linear Inverse Problems." IEEE Transactions on Signal Processing, 2017.
- He H, et al. "Model-Driven Deep Learning for Physical Layer Communications." IEEE Wireless Communications, 2019.
