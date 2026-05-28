import cupy as cp
from layers import Layer, Conv2D, ReLU, MaxPool, Linear, SoftmaxCrossEntropy, Dropout


class MLP:
    def __init__(self, layerlist, dropout=True):
        self.layers = []
        num_in = layerlist[0]
        self.loss_layer = SoftmaxCrossEntropy()
        for num_out in layerlist[1:-1]:
            self.layers.append(Linear(num_in, num_out))
            self.layers.append(ReLU())
            if dropout:
                self.layers.append(Dropout())
            num_in = num_out
        self.layers.append(Linear(num_in, layerlist[-1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad


class CNN:
    def __init__(self):
        self.layers = [
            Conv2D(1, 16, 5, stride=1, pad=2),
            ReLU(),
            MaxPool(2, 2),
            Conv2D(16, 32, 3, stride=1, pad=1),
            ReLU(),
            Conv2D(32, 64, 3, stride=1, pad=1),
            ReLU(),
            MaxPool(2, 2),
        ]
        self.fc = MLP([64*7*7, 256, 26])
        self.all_layers = self.layers + self.fc.layers
        self.loss_layer = SoftmaxCrossEntropy()

    def forward(self, x):
        x = x.reshape(-1, 1, 28, 28)
        for layer in self.layers:
            x = layer.forward(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc.forward(x)

    def loss(self, x, t):
        logits = self.forward(x)
        return self.loss_layer.forward(logits, t)

    def backward(self):
        grad = self.loss_layer.backward()
        grad = self.fc.backward(grad)
        grad = grad.reshape(-1, 64, 7, 7)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train(self):
        for layer in self.all_layers:
            if hasattr(layer, 'train'):
                layer.train()

    def eval(self):
        for layer in self.all_layers:
            if hasattr(layer, 'eval'):
                layer.eval()

    def evaluate(self, x, t):
        logits = self.forward(x)
        loss = self.loss_layer.forward(logits, t)
        pred = cp.argmax(logits, axis=1)
        label = cp.argmax(t, axis=1)
        acc = cp.mean(pred == label)
        return loss, acc
