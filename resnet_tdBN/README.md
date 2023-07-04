#README

### **复现Going Deeper With Directly-Trained Larger Spiking Neural Networks中的Deep Spiking Residual Network**

##layers.py:
实现网络模型中需要的各种层。
包括产生脉冲的函数，膜电位更新公式，在普通层添加时空域的层，tdBN，代替Relu的LIFSpike层。

##model.py
实现网络模型。

##MNIST.py
训练并测试MNIST数据集

##cifar10.py
训练并测试cifar10数据集

