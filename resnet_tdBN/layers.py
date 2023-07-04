import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

#settings
STEPS = 2
DT = 5                #时间步长
SIMWIN = DT * STEPS   #仿真时间窗口
ALPHA = 0.5
VTH = 0.2
TAU = 0.25

alpha = ALPHA

#产生脉冲的函数，并使用替代导数进行近似
class SpikeAct(torch.autograd.Function):
    """
    Implementation of the spiking activation function with an approximation of graient
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.gt(input, 0)

        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        hu = abs(input) < alpha
        hu = hu.float() / (2 * alpha)

        return grad_input * hu

#膜电位更新函数
def state_update(u_t_n1, o_t_n1, W_mul_o_t1_n):
    u_t1_n1 = TAU * u_t_n1 * (1 - o_t_n1) + W_mul_o_t1_n
    o_t1_n1 = SpikeAct.apply(u_t1_n1 - VTH)

    return u_t1_n1, o_t1_n1

#将普通层转换到时间域
class tdLayer(nn.Module):
    """
    Converts a common layer to the time domain.
    The input tensor needs to have an additional time dimension,
    which in this case is on the last dimension of the data.
    When forwarding, a normal layer forward is performed for each time step of the data in that time dimension.

    Args:
        layer (nn.Module):
            The layer needs to convert
        bn (nn.Module):
            If batch-normalization is needed, the BN layer should be passed in together as a parameter.
    """

    def __init__(self, layer, bn=None, steps=STEPS):
        super(tdLayer, self).__init__()
        self.layer = layer
        self.bn = bn
        self.steps = steps

    def forward(self, x):
        x_ = torch.zeros(self.layer(x[..., 0]).shape + (self.steps,), device=x.device)
        # x[..., 0].shape:[128, 3, 32, 32]
        # x_.shape:[128, 3, 32, 32, 2]
        for step in range(self.steps):
            x_[..., step] = self.layer(x[..., step])

        if self.bn is not None:
            x_ = self.bn(x_)

        return x_

#代替ANN版本resnet中的relu激活函数
class LIFSpike(nn.Module):

    def __init__(self, steps=STEPS):
        super(LIFSpike, self).__init__()
        self.steps = steps

    def forward(self, x):
        u = torch.zeros(x.shape[:-1], device=x.device)
        out = torch.zeros(x.shape, device=x.device)

        for step in range(self.steps):
            u, out[..., step] = state_update(u, out[..., max(step - 1, 0)], x[..., step])

        return out

#tdBN 代替ANN版本resnet的BN
class tdBatchNorm(nn.BatchNorm2d):
    """
    在BN时，同时在时间域上做平均。
    Args:
        num_features (int): same with nn.BatchNorm2d
        eps (float): same with nn.BatchNorm2d
        momentum (float): same with nn.BatchNorm2d
        alpha (float): an addtional parameter which may change in resblock.
        affine (bool): same with nn.BatchNorm2d
        track_running_stats (bool): same with nn.BatchNorm2d
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True):
        super(tdBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, input):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        # cumulate running estimates
        if self.training:
            mean = input.mean([0, 2, 3, 4])
            # use biased var in train
            var = input.var([0, 2, 3, 4], unbiased=False)
            n = input.numel() / input.size(1)  # numel: Returns the total number of elements in the input tensor.
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var

        else:
            mean = self.running_mean
            var = self.running_var

        input = self.alpha * VTH * (input - mean[None, :, None, None, None]) / \
                (torch.sqrt(var[None, :, None, None, None] + self.eps))

        if self.affine:
            input = input * self.weight[None, :, None, None, None] + self.bias[None, :, None, None, None]

        return input