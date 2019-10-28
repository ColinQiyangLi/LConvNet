import tensorflow as tf
import numpy
import pickle
import os
import sys
import pickle

from lconvnet.layers.utils import conv_singular_values_numpy


from .tf_to_pytorch import *

TEST_FILE = 'data/cifar10/cifar-10-batches-py/test_batch'
MODEL_1_FILE = 'checkpoints/qian_models/cifar_l2nnn_noadvtrain.v2.pickle'
MODEL_2_FILE = 'checkpoints/qian_models/cifar_l2nnn_advtrain.v3.pickle'

def loadFile( filename ):
    if filename == '' or os.path.isfile(filename) == False: sys.exit()
    f = open( filename, 'rb')
    size = 32*32*3+1
    entry = pickle.load(f, encoding="latin1")
    images = entry['data'].reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
    labels = [numpy.identity(10)[x] for x in entry['labels']]
    return images, labels

def getBatch( data1, data2, index, bsize ):
    begin = index
    end = index + bsize
    size = len(data1)
    index = end % size
    if end <= size: return data1[begin:end],data2[begin:end],index,False
    batch1 = data1[begin:size]
    batch1.extend(data1[0:index])
    batch2 = data2[begin:size]
    batch2.extend(data2[0:index])
    return batch1,batch2,index,True

def pooling( inlayer ):
    return sqrt(avg_pool(inlayer*inlayer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')) * 2.0

class CIFAR10QianModel(nn.Module):
    def __init__(self, modelfile, normalized=False):
        super().__init__()
        self.checkpt = pickle.load(open( modelfile, 'rb'))
        self.Wconv1 = [self.constant("Wconv1_{}".format(i), self.checkpt[0][i]) for i in range(10)]
        self.bconv1 = [self.constant("bconv1_{}".format(i), self.checkpt[1][i]) for i in range(10)]
        self.Wconv2 = [self.constant("Wconv2_{}".format(i), self.checkpt[2][i]) for i in range(10)]
        self.bconv2 = [self.constant("bconv2_{}".format(i), self.checkpt[3][i]) for i in range(10)]
        self.Wconv3 = [self.constant("Wconv3_{}".format(i), self.checkpt[4][i]) for i in range(10)]
        self.bconv3 = [self.constant("bconv3_{}".format(i), self.checkpt[5][i]) for i in range(10)]
        self.lc1 = len(self.checkpt[6])
        self.W1  = [[self.constant("W1_{}_{}".format(i, ii), self.checkpt[6][i][ii]) for ii in range(10)] for i in range(self.lc1)]
        self.b1  = [[self.constant("b1_{}_{}".format(i, ii), self.checkpt[7][i][ii]) for ii in range(10)] for i in range(self.lc1)]
        self.W2  = [self.constant("W2_{}".format(i), self.checkpt[8][i]) for i in range(10)]
        self.b2  = [self.constant("b2_{}".format(i), self.checkpt[9][i]) for i in range(10)]
        self.normalized = normalized

    def constant(self, name, x):
        self.register_parameter(name, nn.Parameter(torch.from_numpy(x)))
        return getattr(self, name)
    
    def forward(self, x):
        if self.normalized: 
            x = x * 255.
            x = x.permute(0, 2, 3, 1).contiguous()
        layer = resize_image_with_crop_or_pad(x, 28, 28) / 255.
        layer = [conv2d(layer/float(len(self.checkpt[0][0])), self.Wconv1[i], strides=[1,1,1,1], padding='SAME') + self.bconv1[i] for i in range(10)]
        layerf= [relu(-layer[i]) for i in range(10)]
        layer = [relu( layer[i]) for i in range(10)]
        layer = [concat([layer[i], layerf[i]],3) for i in range(10)]
        layer = [conv2d(layer[i]/float(len(self.checkpt[2][0])), self.Wconv2[i], strides=[1,1,1,1],padding='SAME') + self.bconv2[i] for i in range(10)]
        layerf= [relu(-layer[i]) for i in range(10)]
        layer = [relu( layer[i]) for i in range(10)]
        layer = [concat([layer[i], layerf[i]],3) for i in range(10)]
        layer = [pooling(layer[i]) for i in range(10)]
        layer = [conv2d( layer[i]/float(len(self.checkpt[4][0])), self.Wconv3[i], strides=[1,1,1,1],padding='SAME') + self.bconv3[i] for i in range(10)]
        layerf= [relu(-layer[i]) for i in range(10)]
        layer = [relu( layer[i]) for i in range(10)]
        layer = [concat([layer[i], layerf[i]], 3) for i in range(10)]
        layer = [pooling(layer[i]) for i in range(10)]
        layer = [reshape(layer[i], [-1, len(self.checkpt[6][0][0])]) for i in range(10)]
        for i in range(self.lc1):
            layer = [matmul(layer[ii], self.W1[i][ii]) + self.b1[i][ii] for ii in range(10)]
            layerf= [relu(-layer[ii]) for ii in range(10)]
            layer = [relu( layer[ii]) for ii in range(10)]
            layer = [concat([layer[ii],layerf[ii]],1) for ii in range(10)]
        return stack([reshape(matmul(layer[i], self.W2[i]) + self.b2[i], [-1]) for i in range(10)], 1)

def cifar10_qian_model(model_index, normalized):
    if model_index == 3:
        return CIFAR10QianModel(MODEL_1_FILE, normalized)
    return CIFAR10QianModel(MODEL_2_FILE, normalized)


if __name__ == "__main__":
    

    o1 = CIFAR10QianModel(MODEL_1_FILE)
    o2 = CIFAR10QianModel(MODEL_2_FILE)

    testData, testLabl = loadFile(TEST_FILE)
    testSize = len(testData)
    batchsize = 200
    bcount = testSize // batchsize
    sum1 = 0.0
    sum2 = 0.0

    for i in range(bcount):
        xbatch, ybatch, junk1, junk2 = getBatch(testData, testLabl, i*batchsize, batchsize)
        xbatch = xbatch.astype(np.float32)
        ybatch = np.array(ybatch, dtype=np.float32)
        xbatch = torch.from_numpy(xbatch)
        ybatch = torch.from_numpy(ybatch)
        acc1 = float((o1(xbatch).argmax(dim=1) == ybatch.argmax(dim=1)).float().mean())
        acc2 = float((o2(xbatch).argmax(dim=1) == ybatch.argmax(dim=1)).float().mean())

        # [acc1,acc2] = sess.run([accuracy1,accuracy2],{x:xbatch,y:ybatch})
        sum1 += acc1
        sum2 += acc2

    print('model 3 accuracy: ' + repr(sum1/float(bcount)))
    print('model 4 accuracy: ' + repr(sum2/float(bcount)))
