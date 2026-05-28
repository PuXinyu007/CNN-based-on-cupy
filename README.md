CNN —— 基于 CuPy 的 EMNIST 字母识别  
从零实现卷积神经网络，不依赖 PyTorch，深入理解反向传播与 GPU 加速。

卷积层	手写 im2col / col2im，支持 stride / padding  
池化层	MaxPool 正向 + 反向传播  
优化器	SGD + Adam   
正则化	Dropou、学习率衰减  
可视化	Loss/Acc 曲线、逐层特征图（Feature Map）

项目结构  
config.py          # 超参数配置  
layers.py          # 网络层定义（Conv2D / MaxPool / Linear / ReLU / Dropout）  
model.py           # CNN 和 MLP 模型架构  
optimizer.py       # SGD / Adam 优化器  
utils.py           # 数据加载、im2col 运算  
visualize.py       # 训练曲线 & 特征图可视化  
train.py           # 训练入口脚本

环境  
Python 3.9+  
CUDA 12.x  
CuPy 12.x  
