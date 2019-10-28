#!/usr/bin/env python3
import tensorflow as tf
import numpy
import pickle
import os
import sys
import pickle

TEST_FILE = 'data/cifar10/cifar-10-batches-py/test_batch'
MODEL_1_FILE = 'lconvnet/external/qian_ext/models/cifar_l2nnn_noadvtrain.v2.pickle'
MODEL_2_FILE = 'lconvnet/external/qian_ext/models/cifar_l2nnn_advtrain.v3.pickle'

def loadFile( filename ):
    if filename == '' or os.path.isfile(filename) == False: sys.exit()
    f = open( filename, 'rb')
    size = 32*32*3+1
    # labels = []
    # images = []
    # for i in range(10000):
    #     arr = numpy.fromstring(f[i*size:(i+1)*size],dtype=numpy.uint8)
    #     labels.append( numpy.identity(10)[arr[0]] )
    #     images.append( arr[1:].reshape((3,32,32)).transpose((1,2,0)))
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
    return 2.0*tf.sqrt(tf.nn.avg_pool(inlayer*inlayer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME'))

def inference( x, modelfile ):
    checkpt = pickle.load( open( modelfile, 'rb' ) )
    Wconv1 = [tf.constant(checkpt[0][i]) for i in range(10)]
    bconv1 = [tf.constant(checkpt[1][i]) for i in range(10)]
    Wconv2 = [tf.constant(checkpt[2][i]) for i in range(10)]
    bconv2 = [tf.constant(checkpt[3][i]) for i in range(10)]
    Wconv3 = [tf.constant(checkpt[4][i]) for i in range(10)]
    bconv3 = [tf.constant(checkpt[5][i]) for i in range(10)]
    lc1 = len(checkpt[6])
    W1  = [[tf.constant(checkpt[6][i][ii]) for ii in range(10)] for i in range(lc1)]
    b1  = [[tf.constant(checkpt[7][i][ii]) for ii in range(10)] for i in range(lc1)]
    W2  = [tf.constant(checkpt[8][i]) for i in range(10)]
    b2  = [tf.constant(checkpt[9][i]) for i in range(10)]
    layer = tf.image.resize_image_with_crop_or_pad(x,28,28) / 255.
    layer = [tf.nn.conv2d(    layer/float(len(checkpt[0][0])), Wconv1[i], strides=[1,1,1,1],padding='SAME') + bconv1[i] for i in range(10)]
    layerf= [tf.nn.relu(-layer[i]) for i in range(10)]
    layer = [tf.nn.relu( layer[i]) for i in range(10)]
    layer = [tf.concat([layer[i],layerf[i]],3) for i in range(10)]
    layer = [tf.nn.conv2d( layer[i]/float(len(checkpt[2][0])), Wconv2[i], strides=[1,1,1,1],padding='SAME') + bconv2[i] for i in range(10)]
    layerf= [tf.nn.relu(-layer[i]) for i in range(10)]
    layer = [tf.nn.relu( layer[i]) for i in range(10)]
    layer = [tf.concat([layer[i],layerf[i]],3) for i in range(10)]
    layer = [pooling(layer[i]) for i in range(10)]
    layer = [tf.nn.conv2d( layer[i]/float(len(checkpt[4][0])), Wconv3[i], strides=[1,1,1,1],padding='SAME') + bconv3[i] for i in range(10)]
    layerf= [tf.nn.relu(-layer[i]) for i in range(10)]
    layer = [tf.nn.relu( layer[i]) for i in range(10)]
    layer = [tf.concat([layer[i],layerf[i]],3) for i in range(10)]
    layer = [pooling(layer[i]) for i in range(10)]
    layer = [tf.reshape(layer[i], [-1,len(checkpt[6][0][0])]) for i in range(10)]
    for i in range(lc1):
        layer = [tf.matmul(layer[ii],W1[i][ii])+b1[i][ii] for ii in range(10)]
        layerf= [tf.nn.relu(-layer[ii]) for ii in range(10)]
        layer = [tf.nn.relu( layer[ii]) for ii in range(10)]
        layer = [tf.concat([layer[ii],layerf[ii]],1) for ii in range(10)]
    return tf.stack([tf.reshape(tf.matmul(layer[i],W2[i])+b2[i],[-1]) for i in range(10)],1)

x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None, 10])
o1 = inference( x, MODEL_1_FILE )
o2 = inference( x, MODEL_2_FILE )
accuracy1= tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(o1,1)),tf.float32))
accuracy2= tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(o2,1)),tf.float32))
sess = tf.Session()
testData, testLabl = loadFile( TEST_FILE )
testSize = len(testData)
batchsize = 200
bcount = testSize//batchsize
sum1 = 0.0
sum2 = 0.0
for i in range(bcount):
    xbatch,ybatch,junk1,junk2 = getBatch( testData, testLabl, i*batchsize, batchsize)
    [acc1,acc2] = sess.run([accuracy1,accuracy2],{x:xbatch,y:ybatch})
    sum1 += acc1
    sum2 += acc2
print('model 3 accuracy: ' + repr(sum1/float(bcount)))
print('model 4 accuracy: ' + repr(sum2/float(bcount)))
