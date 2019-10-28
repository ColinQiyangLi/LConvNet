import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv2d(x, f, strides=None, padding=None):
    h, w, ci, co = f.shape
    assert padding in ["SAME"]
    return F.conv2d(x.permute(0, 3, 1, 2), f.permute(3, 2, 0, 1).contiguous(), 
        stride=strides[1:3], padding=(h // 2, w // 2)).permute(0, 2, 3, 1)

def relu(x):
    return F.relu(x)
    
def concat(x, dim):
    return torch.cat(x, dim=dim)

def stack(x, dim):
    return torch.stack(x, dim=dim)

def sqrt(x):
    return torch.sqrt(x)

def avg_pool(x, ksize, strides, padding):
    assert len(ksize) == 4 and len(strides) == 4 and padding == 'SAME'
    return F.avg_pool2d(x.permute(0, 3, 1, 2), kernel_size=ksize[1:3], 
        stride=strides[1:3]).permute(0, 2, 3, 1)

def matmul(x, weight):
    return F.linear(x, weight.transpose(1, 0).contiguous())

def reshape(x, shape):
    return x.reshape(*shape)

def resize_image_with_crop_or_pad(x, target_height, target_width):
    target = (target_height, target_width)
    _, h, w, _ = x.shape
    assert h - target[0] >= 0 and (h - target[0]) % 2 == 0
    assert w - target[1] >= 0 and (w - target[1]) % 2 == 0
    h = (h - target[0]) // 2
    w = (w - target[1]) // 2
    return x[:, h:-h, w:-w]

if __name__ == "__main__":
    b, h, w, ci, co = 4, 3, 3, 6, 3
    x_tf = np.random.randn(b, 32, 32, ci).astype(np.float32)
    x_torch = TFTensor(torch.from_numpy(x_tf).float())
    f = np.random.randn(h, w, ci, co).astype(np.float32)
    fl = np.random.randn(ci, co).astype(np.float32)

    tf.enable_eager_execution()

    # y_torch = conv2d(x_torch, f, strides=[1, 1, 1, 1], padding='SAME')
    # y_tf = tf.nn.conv2d(x_tf, f, strides=[1, 1, 1, 1], padding='SAME')

    # y_torch = avg_pool(x_torch, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # y_tf = tf.nn.avg_pool(x_tf, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    # y_torch = sqrt(x_torch * x_torch + x_torch * x_torch)
    # y_tf = tf.sqrt(x_tf * x_tf + x_tf * x_tf)

    # y_torch = concat((x_torch, x_torch), 3)
    # y_tf = tf.concat((x_tf, x_tf), 3)

    # y_torch = matmul(reshape(x_torch, [-1, ci]), fl)
    # y_tf = tf.matmul(tf.reshape(x_tf, [-1, ci]), fl)

    y_torch = resize_image_with_crop_or_pad(x_torch, 28, 28) / 255.
    y_tf = tf.image.resize_image_with_crop_or_pad(x_tf, 28, 28) / 255.

    y_torch = y_torch().detach().numpy()
    y_tf = y_tf.numpy()

    print(np.abs(y_torch - y_tf).max())
