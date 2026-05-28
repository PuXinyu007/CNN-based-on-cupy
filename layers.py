import cupy as cp
from config import config, eps
from utils import im2col, col2im

class Layer:
    def __init__(self):
        self.training = True

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad=0):
        self.C_out = out_channels
        self.C_in = in_channels
        self.K = kernel_size
        self.stride = stride
        self.pad = pad
        self.W = cp.random.randn(out_channels, in_channels, kernel_size, kernel_size) * cp.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.b = cp.zeros((1, out_channels))

    def forward(self, x):
        self.x = x
        N = x.shape[0]
        self.col, self.cache_shape = im2col(x, self.K, self.K, self.stride, self.pad)
        W_col = self.W.reshape(self.C_out, -1)
        out = W_col @ self.col + self.b.T 
        H_out = self.cache_shape[-2]
        W_out = self.cache_shape[-1]
        out = out.reshape(self.C_out, H_out, W_out, N).transpose(3, 0, 1, 2)
        return out

    def backward(self, grad):
        N = grad.shape[0]
        H_out, W_out = grad.shape[2], grad.shape[3]
        grad_reshaped = grad.transpose(1, 2, 3, 0).reshape(self.C_out, -1)
        self.grad_W = grad_reshaped @ self.col.T
        self.grad_W = self.grad_W.reshape(self.W.shape)
        self.grad_b = cp.sum(grad_reshaped, axis=1, keepdims=True).T
        W_col = self.W.reshape(self.C_out, -1)
        dcol = W_col.T @ grad_reshaped
        grad = col2im(dcol, self.x.shape, self.K, self.K, self.stride, self.pad)
        return grad


class MaxPool(Layer):
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        H_out = H // self.pool_size
        W_out = W // self.pool_size
        x_reshaped = x.reshape(N, C, H_out, self.pool_size, W_out, self.pool_size)
        x_reshaped = x_reshaped.transpose(0, 1, 2, 4, 3, 5).reshape(-1, self.pool_size * self.pool_size)
        self.max_idx = cp.argmax(x_reshaped, axis=1)
        out = x_reshaped[cp.arange(x_reshaped.shape[0]), self.max_idx]
        return out.reshape(N, C, H_out, W_out)

    def backward(self, grad):
        N, C, H_out, W_out = grad.shape
        H, W = self.x.shape[2], self.x.shape[3]
        grad_flat = grad.reshape(-1)
        dx_reshaped = cp.zeros((N * C * H_out * W_out, self.pool_size * self.pool_size))
        dx_reshaped[cp.arange(dx_reshaped.shape[0]), self.max_idx] = grad_flat
        dx = dx_reshaped.reshape(N, C, H_out, W_out, self.pool_size, self.pool_size)
        dx = dx.transpose(0, 1, 2, 4, 3, 5).reshape(N, C, H, W)
        return dx


class Linear(Layer):
    def __init__(self, num_in, num_out):
        self.num_in = num_in
        self.num_out = num_out
        self.W = cp.random.randn(num_in, num_out) * cp.sqrt(2 / num_in)
        self.b = cp.zeros((1, num_out))

    def forward(self, x):
        self.x = x
        self.y = x @ self.W + self.b
        return self.y

    def backward(self, grad):
        self.grad_W = self.x.T @ grad
        self.grad_b = cp.sum(grad, axis=0, keepdims=True)
        grad = grad @ self.W.T
        return grad


class ReLU(Layer):
    def forward(self, x):
        self.x = x
        self.y = cp.maximum(x, 0)
        return self.y

    def backward(self, grad):
        return grad * (self.x > 0)


class SoftmaxCrossEntropy(Layer):
    def forward(self, x, t):
        self.t = t
        x_max = cp.max(x, axis=1, keepdims=True)
        exp_x = cp.exp(x - x_max)
        self.y = exp_x / cp.sum(exp_x, axis=1, keepdims=True) 
        loss = -cp.sum(self.t * cp.log(self.y + eps)) / self.y.shape[0]
        return loss

    def backward(self, grad=None):
        return (self.y - self.t) / float(self.y.shape[0])


class Dropout(Layer):
    def __init__(self):
        super().__init__()
        self.drop_rate = config['drop_rate']
        self.mask = None

    def forward(self, x):
        if not self.training:
            return x
        keep_prob = 1 - self.drop_rate
        self.mask = (cp.random.rand(*x.shape) < keep_prob).astype(x.dtype)
        return x * self.mask / keep_prob

    def backward(self, grad):
        if self.mask is None:
            return grad
        return grad * self.mask / (1 - self.drop_rate)
