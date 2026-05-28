import os
import cupy as cp
import numpy as np
import pandas as pd


def im2col(x, kernel_h, kernel_w, stride=1, pad=0):
    N, C, H, W = x.shape
    H_out = (H + 2 * pad - kernel_h) // stride + 1
    W_out = (W + 2 * pad - kernel_w) // stride + 1

    if pad > 0:
        x = cp.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

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


def load_emnist_csv(file_path, num_classes=26):
    data_cpu = pd.read_csv(file_path, header=None).values
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
