import tensorflow as tf
import tensorflow.contrib.layers as lays
import os
import numpy as np
import tables
import cv2


channels = 3
rows = 64
cols = 64
img_dtype = tables.Float32Atom()
numPersons = 625

filter1 = 8
filter2 = 16
filter3 = 32
filter4 = 64
filter5 = 128
filterSize = 5
stride = [1,1,1,1]
padding = 'SAME'
maxpoolSize = [1,2,2,1]
poolStride = [1,2,2,1]
poolPadding = 'VALID'

convResult = 1024
fc1Units = 768
fc2Units = numPersons

graph = tf.Graph()
with graph.as_default():

    WC1 = tf.Variable(tf.truncated_normal([filterSize, filterSize, channels, filter1], stddev=0.1), dtype=tf.float32)
    bC1 = tf.Variable(tf.zeros([filter1]), dtype=tf.float32)

    WC2 = tf.Variable(tf.truncated_normal([filterSize, filterSize, filter1, filter2],stddev=0.1), dtype=tf.float32)
    bC2 = tf.Variable(tf.zeros([filter2]), dtype=tf.float32)

    WC3 = tf.Variable(tf.truncated_normal([filterSize, filterSize, filter2, filter3],stddev=0.1), dtype=tf.float32)
    bC3 = tf.Variable(tf.zeros([filter3]), dtype=tf.float32)

    WC4 = tf.Variable(tf.truncated_normal([filterSize, filterSize, filter3, filter4],stddev=0.1), dtype=tf.float32)
    bC4 = tf.Variable(tf.zeros([filter4]), dtype=tf.float32)

    Wf1 = tf.Variable(tf.truncated_normal([convResult,fc1Units],stddev=0.1), dtype=tf.float32)
    bf1 = tf.Variable(tf.zeros([fc1Units]), dtype=tf.float32)

    Wf2 = tf.Variable(tf.truncated_normal([fc1Units,fc2Units],stddev=0.1), dtype=tf.float32)
    bf2 = tf.Variable(tf.zeros([fc2Units]), dtype=tf.float32)

    def CNN(frames):

        conv = tf.nn.relu(tf.nn.conv2d(input=frames, filter=WC1, strides=stride, padding=padding) + bC1)
        conv = tf.nn.max_pool(conv, ksize=maxpoolSize, strides=poolStride, padding=poolPadding)

        conv = tf.nn.relu(tf.nn.conv2d(input=conv, filter=WC2, strides=stride, padding=padding) + bC2)
        conv = tf.nn.max_pool(conv, ksize=maxpoolSize, strides=poolStride, padding=poolPadding)

        conv = tf.nn.relu(tf.nn.conv2d(input=conv, filter=WC3, strides=stride, padding=padding) + bC3)
        conv = tf.nn.max_pool(conv, ksize=maxpoolSize, strides=poolStride, padding=poolPadding)

        conv = tf.nn.relu(tf.nn.conv2d(input=conv, filter=WC4, strides=stride, padding=padding) + bC4)
        conv = tf.nn.max_pool(conv, ksize=maxpoolSize, strides=poolStride, padding=poolPadding)

        return conv


    def getFeatureVector(inputs):

        conv = CNN(inputs)
        conv = lays.flatten(conv)

        fc1 = tf.nn.relu(tf.matmul(conv, Wf1) + bf1)
        fc1 = tf.nn.dropout(fc1, tf_keep_prob)
        fc2 = tf.matmul(fc1, Wf2) + bf2

        return fc2


    tf_train_X = tf.placeholder(tf.float32, [None,rows,cols,channels])
    tf_train_Y = tf.placeholder(tf.float32, [None, numPersons])
    tf_keep_prob = tf.placeholder_with_default(1.0, ())

    logits = getFeatureVector(tf_train_X)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_Y))

    optimizer = tf.train.AdamOptimizer().minimize(loss)
    tf_prediction = tf.nn.softmax(logits)
    saver = tf.train.Saver()


def saveFeatureVectors(step, path, savePath):

    hdf5_file = tables.open_file(savePath, mode='w')
    persons = os.listdir(path)
    numPersons = len(persons)
    frames = np.ndarray([numPersons,rows,cols,channels], np.float32)
    features_array = hdf5_file.create_earray(hdf5_file.root, 'features', img_dtype, shape=(0, 625))

    for per in range(numPersons):

        frame = cv2.imread(os.path.join(path,persons[per]))
        frame = cv2.resize(frame, (rows, cols)) / 255
        frames[per,:,:,:] = frame
        print(persons[per])

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        saver.restore(session, 'models/model-' + str(step))

        features = session.run([logits], feed_dict={tf_train_X: frames})[0]
        [features_array.append(f[None]) for f in features]

    hdf5_file.close()


step = 4
path = os.path.join('D:\\','Python','Person Re ID Dataset','VIPeR','cam_a')
savePath = os.path.join('D:\\','Python','Person Re ID Dataset','VIPeR','cam_aCNN.h5')
saveFeatureVectors(step,path,savePath)


path = os.path.join('D:\\','Python','Person Re ID Dataset','VIPeR','cam_b')
savePath = os.path.join('D:\\','Python','Person Re ID Dataset','VIPeR','cam_bCNN.h5')
saveFeatureVectors(step,path,savePath)