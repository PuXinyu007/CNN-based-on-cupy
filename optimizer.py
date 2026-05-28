import cupy as cp
from config import eps


class Optimizer:
    def __init__(self, lr, weight_decay):
        self.lr = lr
        self.weight_decay = weight_decay

    def update(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, model, lr=0.01, weight_decay=5e-4):
        super().__init__(lr, weight_decay)
        self.layers_to_update = model.layers + model.fc.layers

    def update(self):
        for layer in self.layers_to_update:
            if hasattr(layer, 'W'):
                layer.W -= self.lr * (layer.grad_W + self.weight_decay * layer.W)
                layer.b -= self.lr * layer.grad_b


class Adam(Optimizer):
    def __init__(self, model, lr=0.001, weight_decay=5e-4, beta1=0.9, beta2=0.999):
        super().__init__(lr, weight_decay)
        self.layers_to_update = model.layers + model.fc.layers
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0

        self.m = {}
        self.v = {}

        for layer in self.layers_to_update:
            if hasattr(layer, 'W'):
                self.m[layer] = {'W': cp.zeros_like(layer.W), 'b': cp.zeros_like(layer.b)}
                self.v[layer] = {'W': cp.zeros_like(layer.W), 'b': cp.zeros_like(layer.b)}

    def update(self):
        self.t += 1
        for layer in self.layers_to_update:
            if hasattr(layer, 'W'):
                m_W, v_W = self.m[layer]['W'], self.v[layer]['W']
                m_b, v_b = self.m[layer]['b'], self.v[layer]['b']

                grad_W = layer.grad_W
                grad_b = layer.grad_b
                m_W = self.beta1 * m_W + (1 - self.beta1) * grad_W
                v_W = self.beta2 * v_W + (1 - self.beta2) * (grad_W ** 2)
                m_b = self.beta1 * m_b + (1 - self.beta1) * grad_b
                v_b = self.beta2 * v_b + (1 - self.beta2) * (grad_b ** 2)

                m_hat_W = m_W / (1 - self.beta1 ** self.t)
                v_hat_W = v_W / (1 - self.beta2 ** self.t)
                m_hat_b = m_b / (1 - self.beta1 ** self.t)
                v_hat_b = v_b / (1 - self.beta2 ** self.t)

                layer.W -= self.lr * (m_hat_W / (cp.sqrt(v_hat_W) + eps) + self.weight_decay * layer.W)
                layer.b -= self.lr * m_hat_b / (cp.sqrt(v_hat_b) + eps)

                self.m[layer]['W'], self.v[layer]['W'] = m_W, v_W
                self.m[layer]['b'], self.v[layer]['b'] = m_b, v_b
