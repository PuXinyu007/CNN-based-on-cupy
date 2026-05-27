import cupy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

config = {
    'lr': 0.001, 
    'weight_decay': 5e-4,
    'drop_rate': 0.5,
    'num_epochs': 20,
    'batch_size': 256,
    'milestones': [10, 15],
    'seed': 0,
    'data_path': r"E:\Code\data\emnist-letters-train.csv"
}

eps = 1e-8
lr_decay = 0.1

cp.random.seed(config['seed'])

def im2col(x, kernel_h, kernel_w, stride = 1, pad = 0):
    N, C, H, W = x.shape
    H_out = (H + 2 * pad - kernel_h) // stride + 1
    W_out = (W + 2 * pad - kernel_w) // stride + 1

    if pad > 0:
        x = cp.pad(x,((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

    col = cp.zeros((C * kernel_h * kernel_w, N * H_out * W_out))

    idx = 0
    for i in range(0, H_out * stride, stride):
        for j in range(0, W_out * stride, stride):
            patch = x[:, :, i:i+kernel_h, j:j+kernel_w] 
            col[:, idx:idx+N] = patch.reshape(N, -1).T
            idx += N

    return col, (N, C, H, W, kernel_h, kernel_w, stride, pad, H_out, W_out)

def col2im(col, x_shape, kernel_h, kernel_w, stride=1, pad=0):
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2*pad, W + 2*pad
    dx_padded = cp.zeros((N, C, H_padded, W_padded))

    H_out = (H_padded - kernel_h) // stride + 1
    W_out = (W_padded - kernel_w) // stride + 1

    idx = 0
    for i in range(0, H_out * stride, stride):
        for j in range(0, W_out * stride, stride):
            patch = col[:, idx:idx+N].T.reshape(N, C, kernel_h, kernel_w)
            dx_padded[:, :, i:i+kernel_h, j:j+kernel_w] += patch
            idx += N

    if pad > 0:
        dx = dx_padded[:, :, pad:-pad, pad:-pad]
    else:
        dx = dx_padded

    return dx

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
        # 按 (C_out, H_out, W_out, N) 重组，再转置为 (N, C_out, H_out, W_out)
        out = out.reshape(self.C_out, H_out, W_out, N).transpose(3, 0, 1, 2)
        return out

    def backward(self, grad):
        N = grad.shape[0]
        H_out, W_out = grad.shape[2], grad.shape[3]
        # 把 grad 展平成与 col 列顺序一致
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
            return x                 # 推理时原样通过
        
        keep_prob = 1 - self.drop_rate
        self.mask = (cp.random.rand(*x.shape) < keep_prob).astype(x.dtype)
        return x * self.mask / keep_prob   # inverted dropout

    def backward(self, grad):
        return grad * self.mask / (1 - self.drop_rate)

class MLP:
    def __init__(self, layerlist, dropout=True):
        self.layers = []
        num_in = layerlist[0]
        self.loss_layer = SoftmaxCrossEntropy()
        for num_out in layerlist[1:-1]:
            self.layers.append(Linear(num_in, num_out))
            self.layers.append(ReLU())
            if dropout:
                self.layers.append(Dropout())  # 每个隐藏层后面跟一个
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
            if hasattr(layer,'W'):
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
        
model = CNN()
optimizer = Adam(model, lr=config['lr'], weight_decay=config['weight_decay'])

def load_emnist_csv(file_path, num_classes=26):
    print("正在从硬盘加载数据...")
    data_cpu = pd.read_csv(file_path, header=None).values

    print("正在将数据传输至 GPU 显存，请稍候...")
    data = cp.asarray(data_cpu)

    y = data[:, 0]
    x = data[:, 1:]

    labels_0_indexed = (y - 1).astype(cp.int32)
    t = cp.eye(num_classes)[labels_0_indexed]

    N = x.shape[0]
    images_reshaped = x.reshape(N, 28, 28)
    images_corrected = images_reshaped.transpose(0, 2, 1) 
    x = images_corrected.reshape(N, 784) / 255.0

    print(f"样本量: {x.shape[0]}, 特征维度: {x.shape[1]}")
    return x, t

X, T = load_emnist_csv(config['data_path'], num_classes=26)
ratio = 0.95
split = int(ratio * X.shape[0])
idx = cp.random.permutation(X.shape[0])
X = X[idx]
T = T[idx]
X_train, X_test = X[0:split], X[split:]
T_train, T_test = T[0:split], T[split:]

train_losses = []
train_accs = []
test_losses = []
test_accs = []
batch_size = config['batch_size']
num_epochs = config['num_epochs']
milestones = config['milestones']
num_samples = X_train.shape[0]

for epoch in range(num_epochs):
    shuffled_indices = cp.random.permutation(num_samples)
    X_train_s = X_train[shuffled_indices]
    T_train_s = T_train[shuffled_indices]

    epoch_loss = 0
    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, num_samples)

        x_batch = X_train_s[start_idx:end_idx]
        t_batch = T_train_s[start_idx:end_idx]

        model.train()
        loss = model.loss(x_batch, t_batch)
        epoch_loss += float(loss) * (end_idx - start_idx)

        model.backward()
        optimizer.update()

    if epoch + 1 in milestones:
        optimizer.lr *= lr_decay   
    epoch_loss /= num_samples

    model.eval()
    _, train_acc = model.evaluate(X_train[:2000], T_train[:2000])
    train_acc = float(train_acc)

    test_loss, test_acc = model.evaluate(X_test, T_test)
    test_loss = float(test_loss)
    test_acc = float(test_acc) 
    
    train_losses.append(epoch_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {epoch_loss:.4f} - Test Acc: {test_acc*100:.2f}%")


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)  
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', color='crimson', linewidth=2)
plt.plot(range(1, num_epochs + 1), test_losses, color='blue', ls='--', label='Test Loss', linewidth=2)
plt.title('Loss over Epochs', fontsize=12)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6) 
plt.legend()

plt.subplot(1, 2, 2)
acc_percentage = [a * 100 for a in train_accs] 
plt.plot(range(1, num_epochs + 1), acc_percentage, label='Train Accuracy', color='red', linewidth=2)
acc_percentage = [a * 100 for a in test_accs] 
plt.plot(range(1, num_epochs + 1), acc_percentage, label='Test Accuracy', color='royalblue', linewidth=2)
plt.title('Accuracy over Epochs', fontsize=12)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Accuracy (%)', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()



def visualize_all_channels(model, single_image):
    """
    输入单张图片，将网络中每一个卷积/池化层的所有通道以网格(Grid)形式完整画出来
    """
    # 确保输入是 4D 张量: (1, 1, 28, 28)
    if len(single_image.shape) == 3:
        current_x = single_image.reshape(1, *single_image.shape)
    elif len(single_image.shape) == 2:
        current_x = single_image.reshape(1, 1, *single_image.shape)
    else:
        current_x = single_image
        
    for i, layer in enumerate(model.layers):
        current_x = layer.forward(current_x)
        layer_name = layer.__class__.__name__
        
        if len(current_x.shape) == 4:
            # 拿到当前的通道数 (例如 16, 32, 64)
            num_channels = current_x.shape[1] 
            
            # 将 GPU 数据传回 CPU 并去掉 Batch 维度 -> 变成 (Channel, H, W)
            feat_maps = current_x[0].get() if hasattr(current_x, 'get') else current_x[0]
            
            # 自动计算网格的行数和列数（固定每行放 8 张小图）
            cols = min(8, num_channels)
            rows = int(np.ceil(num_channels / cols))
            
            # 创建子图画布
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
            fig.suptitle(f"Layer {i+1}: {layer_name} | Channels: {num_channels} | Shape: {feat_maps.shape[1:]}", 
                         fontsize=12, fontweight='bold', y=0.98)
            
            # 如果只有一行或一列，axes 会是一维数组，统一扁平化方便循环处理
            if num_channels == 1:
                axes = np.array([axes])
            else:
                axes = axes.flatten()
                
            # 循环把每一个通道的特征图画进网格
            for c in range(num_channels):
                ax = axes[c]
                # 使用 'viridis' 渲染，特征强烈的像素会发黄，微弱的会变紫，对比度极高
                ax.imshow(feat_maps[c], cmap='viridis')
                ax.axis('off')
                ax.set_title(f"Ch {c}", fontsize=8)
                
            # 如果网格格子比通道数多（比如26个通道，排了4x8=32个格子），把剩下的空白格子隐藏掉
            for extra in range(num_channels, len(axes)):
                axes[extra].axis('off')
                
            plt.tight_layout()
            plt.show()
            
        elif layer_name == "ReLU":
            pass
        
        
sample_images = X_test[0] 
# Conv2D 需要 4D 输入：[Batch, Channel, Height, Width]
sample_images_4d = sample_images.reshape(1, 1, 28, 28)
model.eval()       
visualize_all_channels(model, sample_images_4d)