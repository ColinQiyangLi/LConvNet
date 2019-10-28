import tensorflow as tf
import numpy
import pickle
import os
import sys

from .tf_to_pytorch import *

TEST_IMAGE_FILE  = 'data/mnist/MNIST/raw/t10k-images-idx3-ubyte'
TEST_LABEL_FILE  = 'data/mnist/MNIST/raw/t10k-labels-idx1-ubyte'
MODEL_1_FILE = 'checkpoints/qian_models/mnist_l2nnn_noadvtrain.v2.pickle'
MODEL_2_FILE = 'checkpoints/qian_models/mnist_l2nnn_advtrain.v2.pickle'

def readImages( filename ):
    if filename == '' or os.path.isfile(filename) == False: sys.exit()
    file = open( filename, "rb" )
    u32 = numpy.dtype(numpy.uint32).newbyteorder('>')
    magic = numpy.frombuffer(file.read(4), dtype=u32)[0]
    if magic != 2051: sys.exit()
    count = numpy.frombuffer(file.read(4), dtype=u32)[0]
    dim1 = numpy.frombuffer(file.read(4), dtype=u32)[0]
    if dim1 != 28: sys.exit()
    dim2 = numpy.frombuffer(file.read(4), dtype=u32)[0]
    if dim2 != 28: sys.exit()
    data = numpy.frombuffer(file.read(count*dim1*dim2),dtype=numpy.uint8).reshape(count,dim1*dim2)
    file.close()
    return data

def readLabels( filename ):
    if filename == '' or os.path.isfile(filename) == False: sys.exit()
    file = open( filename, "rb" )
    u32 = numpy.dtype(numpy.uint32).newbyteorder('>')
    magic = numpy.frombuffer(file.read(4), dtype=u32)[0]
    if magic != 2049: sys.exit()
    count = numpy.frombuffer(file.read(4), dtype=u32)[0]
    data = numpy.frombuffer( file.read(count), dtype=numpy.uint8)
    file.close()
    return data

def pooling( inlayer ):
    return 2.0*sqrt(avg_pool(inlayer*inlayer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME'))

class MNISTQianModel(nn.Module):
    def __init__(self, modelfile, normalized=False):
        super().__init__()
        self.checkpt = pickle.load( open( modelfile, 'rb' ) )
        self.Wconv1 = self.constant("Wconv1_0", self.checkpt[0])
        self.bconv1 = self.constant("bconv1_0", self.checkpt[1])
        self.Wconv2 = self.constant("Wconv2_0", self.checkpt[2])
        self.bconv2 = self.constant("bconv2_0", self.checkpt[3])
        self.lc1 = len(self.checkpt[4])
        self.W1  = [self.constant("W1_{}".format(i), self.checkpt[4][i]) for i in range(self.lc1)]
        self.b1  = [self.constant("b1_{}".format(i), self.checkpt[5][i]) for i in range(self.lc1)]
        self.lc2 = len(self.checkpt[6])
        self.W2  = [[self.constant("W2_{}_{}".format(i, ii), self.checkpt[6][i][ii]) for ii in range(10)] for i in range(self.lc2)]
        self.b2  = [[self.constant("b2_{}_{}".format(i, ii), self.checkpt[7][i][ii]) for ii in range(10)] for i in range(self.lc2)]
        self.W3  = [self.constant("W3_{}".format(i), self.checkpt[8][i]) for i in range(10)]
        self.b3  = [self.constant("b3_{}".format(i), self.checkpt[9][i]) for i in range(10)]
        self.normalized = normalized
        # self.dummy = nn.Parameter(torch.tensor(0.))


    def constant(self, name, x):
        self.register_parameter(name, nn.Parameter(torch.from_numpy(x)))
        return getattr(self, name)

    def forward(self, x):
        if self.normalized: 
            x = x * 255.
            x = x.permute(0, 2, 3, 1).contiguous()
        layer = reshape(x,[-1,28,28,1])/255.
        layer = conv2d(layer/float(len(self.checkpt[0])),self.Wconv1,strides=[1,1,1,1],padding='SAME')+self.bconv1
        layerf= relu(-layer)
        layer = relu( layer)
        layer = pooling(concat([layer,layerf],3))
        layer = conv2d(layer/float(len(self.checkpt[2])),self.Wconv2,strides=[1,1,1,1],padding='SAME')+self.bconv2
        layerf= relu(-layer)
        layer = relu( layer)
        layer = pooling(concat([layer,layerf],1))
        layer = reshape(layer, [-1,len(self.checkpt[4][0])])
        for i in range(self.lc1):
            layer = matmul(layer,self.W1[i])+self.b1[i]
            layerf= relu(-layer)
            layer = relu(layer)
            layer = concat([layer,layerf],1)
        layer = [layer for i in range(10)]
        for i in range(self.lc2):
            layer = [matmul(layer[ii], self.W2[i][ii])+self.b2[i][ii] for ii in range(10)]
            layerf= [relu(-layer[ii]) for ii in range(10)]
            layer = [relu( layer[ii]) for ii in range(10)]
            layer = [concat([layer[ii],layerf[ii]],1) for ii in range(10)]
        return stack([reshape(matmul(layer[i],self.W3[i])+self.b3[i],[-1]) for i in range(10)],1)

def mnist_qian_model(model_index, normalized):
    if model_index == 3:
        return MNISTQianModel(MODEL_1_FILE, normalized)
    return MNISTQianModel(MODEL_2_FILE, normalized)

if __name__ == "__main__":
    
    o1 = MNISTQianModel(MODEL_1_FILE)
    o2 = MNISTQianModel(MODEL_2_FILE)

    testData = readImages( TEST_IMAGE_FILE )
    testLabl = readLabels( TEST_LABEL_FILE ) 

    xbatch = testData.astype(np.float32)
    xbatch = torch.from_numpy(xbatch)
    ybatch = torch.from_numpy(testLabl).long()
    acc1 = float((o1(xbatch).argmax(dim=1) == ybatch).float().mean())
    acc2 = float((o2(xbatch).argmax(dim=1) == ybatch).float().mean())

    print('Model 3 accuracy: ' + repr(acc1))
    print('Model 4 accuracy: ' + repr(acc2))
