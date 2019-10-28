import tensorflow as tf
import numpy
import pickle
import os
import sys

TEST_IMAGE_FILE  = 'data/mnist/MNIST/raw/t10k-images-idx3-ubyte'
TEST_LABEL_FILE  = 'data/mnist/MNIST/raw/t10k-labels-idx1-ubyte'
MODEL_1_FILE = 'lconvnet/external/qian_ext/models/mnist_l2nnn_noadvtrain.v2.pickle'
MODEL_2_FILE = 'lconvnet/external/qian_ext/models/mnist_l2nnn_advtrain.v2.pickle'

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
    return 2.0*tf.sqrt(tf.nn.avg_pool(inlayer*inlayer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME'))

def inference( x, modelfile ):
    checkpt = pickle.load( open( modelfile, 'rb' ) )
    Wconv1 = tf.constant(checkpt[0])
    bconv1 = tf.constant(checkpt[1])
    Wconv2 = tf.constant(checkpt[2])
    bconv2 = tf.constant(checkpt[3])
    lc1 = len(checkpt[4])
    W1  = [tf.constant(checkpt[4][i]) for i in range(lc1)]
    b1  = [tf.constant(checkpt[5][i]) for i in range(lc1)]
    lc2 = len(checkpt[6])
    W2  = [[tf.constant(checkpt[6][i][ii]) for ii in range(10)] for i in range(lc2)]
    b2  = [[tf.constant(checkpt[7][i][ii]) for ii in range(10)] for i in range(lc2)]
    W3  = [tf.constant(checkpt[8][i]) for i in range(10)]
    b3  = [tf.constant(checkpt[9][i]) for i in range(10)]
    layer = tf.reshape(x,[-1,28,28,1])/255.
    layer = tf.nn.conv2d(layer/float(len(checkpt[0])),Wconv1,strides=[1,1,1,1],padding='SAME')+bconv1
    layerf= tf.nn.relu(-layer)
    layer = tf.nn.relu( layer)
    layer = pooling(tf.concat([layer,layerf],3))
    layer = tf.nn.conv2d(layer/float(len(checkpt[2])),Wconv2,strides=[1,1,1,1],padding='SAME')+bconv2
    layerf= tf.nn.relu(-layer)
    layer = tf.nn.relu( layer)
    layer = pooling(tf.concat([layer,layerf],1))
    layer = tf.reshape(layer, [-1,len(checkpt[4][0])])
    for i in range(lc1):
        layer = tf.matmul(layer,W1[i])+b1[i]
        layerf= tf.nn.relu(-layer)
        layer = tf.nn.relu(layer)
        layer = tf.concat([layer,layerf],1)
    layer = [layer for i in range(10)]
    for i in range(lc2):
        layer = [tf.matmul(layer[ii],W2[i][ii])+b2[i][ii] for ii in range(10)]
        layerf= [tf.nn.relu(-layer[ii]) for ii in range(10)]
        layer = [tf.nn.relu( layer[ii]) for ii in range(10)]
        layer = [tf.concat([layer[ii],layerf[ii]],1) for ii in range(10)]
    return tf.stack([tf.reshape(tf.matmul(layer[i],W3[i])+b3[i],[-1]) for i in range(10)],1)

x  = tf.placeholder(tf.float32, [None, 28*28])
y  = tf.placeholder(tf.int64,   [None])
o1 = inference( x, MODEL_1_FILE )
o2 = inference( x, MODEL_2_FILE )
accuracy1= tf.reduce_mean(tf.cast(tf.equal(y,tf.argmax(o1,1)),tf.float32))
accuracy2= tf.reduce_mean(tf.cast(tf.equal(y,tf.argmax(o2,1)),tf.float32))
sess = tf.Session()
import pdb; pdb.set_trace()
testData = readImages( TEST_IMAGE_FILE )
testLabl = readLabels( TEST_LABEL_FILE ) 
[acc1,acc2] = sess.run([accuracy1,accuracy2],{x:testData,y:testLabl})

print('Model 3 accuracy: ' + repr(acc1))
print('Model 4 accuracy: ' + repr(acc2))
