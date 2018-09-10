import tensorflow as tf
import tensorflow.contrib.layers as lays
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

root = os.path.join('D:\\','Python','Person Re ID Dataset','bbox_train', 'bbox_train')
persons = os.listdir(root)
numPersons = len(persons)

rows = 64
cols = 64
channels = 3

filter1 = 16
filter2 = 32
filter3 = 32
filter4 = 64
filter5 = 64
filterSize = 5
stride = [1,1,1,1]
padding = 'SAME'
maxpoolSize = [1,2,2,1]
poolStride = [1,2,2,1]
poolPadding = 'VALID'

convResult = 256
fc1Units = 256
fc2Units = 128

embeddingSize = 128
keep_prob = 0.7
keep_prob_rnn = 0.7

margin = 2
learning_rate = 0.01

isTrain = True

graph = tf.Graph()
with graph.as_default():

    WC1 = tf.get_variable('WC1', [filterSize, filterSize, channels, filter1], tf.float32, lays.xavier_initializer())
    bC1 = tf.get_variable('bC1', [filter1], tf.float32, tf.zeros_initializer())

    WC2 = tf.get_variable('WC2', [filterSize, filterSize, filter1, filter2], tf.float32, lays.xavier_initializer())
    bC2 = tf.get_variable('bC2', [filter2], tf.float32, tf.zeros_initializer())

    WC3 = tf.get_variable('WC3', [filterSize, filterSize, filter2, filter3], tf.float32, lays.xavier_initializer())
    bC3 = tf.get_variable('bC3', [filter3], tf.float32, tf.zeros_initializer())

    WC4 = tf.get_variable('WC4', [filterSize, filterSize, filter3, filter4], tf.float32, lays.xavier_initializer())
    bC4 = tf.get_variable('bC4', [filter4], tf.float32, tf.zeros_initializer())

    WC5 = tf.get_variable('WC5', [filterSize, filterSize, filter4, filter5], tf.float32, lays.xavier_initializer())
    bC5 = tf.get_variable('bC5', [filter5], tf.float32, tf.zeros_initializer())

    Wf1 = tf.get_variable('Wf1', [convResult ,fc1Units], tf.float32, lays.xavier_initializer())
    bf1 = tf.get_variable('bf1', [fc1Units], tf.float32, tf.zeros_initializer())

    Wf2 = tf.get_variable('Wf2', [fc1Units, fc2Units], tf.float32, lays.xavier_initializer())
    bf2 = tf.get_variable('bf2', [fc2Units], tf.float32, tf.zeros_initializer())

    Wf3 = tf.get_variable('Wf3', [embeddingSize, numPersons], tf.float32, lays.xavier_initializer())
    bf3 = tf.get_variable('bf3', [numPersons], tf.float32, tf.zeros_initializer())

    def CNN(frames):

        conv = tf.nn.tanh(tf.nn.conv2d(input=frames, filter=WC1, strides=stride, padding=padding) + bC1)
        conv = tf.nn.max_pool(conv, ksize=maxpoolSize, strides=poolStride, padding=poolPadding)

        conv = tf.nn.tanh(tf.nn.conv2d(input=conv, filter=WC2, strides=stride, padding=padding) + bC2)
        conv = tf.nn.max_pool(conv, ksize=maxpoolSize, strides=poolStride, padding=poolPadding)

        conv = tf.nn.tanh(tf.nn.conv2d(input=conv, filter=WC3, strides=stride, padding=padding) + bC3)
        conv = tf.nn.max_pool(conv, ksize=maxpoolSize, strides=poolStride, padding=poolPadding)

        conv = tf.nn.tanh(tf.nn.conv2d(input=conv, filter=WC4, strides=stride, padding=padding) + bC4)
        conv = tf.nn.max_pool(conv, ksize=maxpoolSize, strides=poolStride, padding=poolPadding)

        conv = tf.nn.tanh(tf.nn.conv2d(input=conv, filter=WC5, strides=stride, padding=padding) + bC5)
        conv = tf.nn.max_pool(conv, ksize=maxpoolSize, strides=poolStride, padding=poolPadding)

        return conv

    def getFeatureVector(inputs):

        conv = CNN(inputs)
        conv = lays.flatten(conv)

        fc1 = tf.nn.tanh(tf.matmul(conv, Wf1) + bf1)
        fc1 = tf.nn.dropout(fc1, tf_keep_prob)
        fc2 = tf.nn.tanh(tf.matmul(fc1, Wf2) + bf2)
        fc2 = tf.nn.dropout(fc2, tf_keep_prob)
        fc3 = tf.matmul(fc2,Wf3) + bf3

        return fc3


    tf_train = tf.placeholder(tf.float32, [None,rows,cols,channels])
    tf_labels = tf.placeholder(tf.float32, [None, numPersons])
    tf_keep_prob = tf.placeholder_with_default(1.0, ())

    logits = getFeatureVector(tf_train)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_labels))
    tf_prediction = tf.argmax(tf.nn.softmax(logits), 1)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    saver = tf.train.Saver()


numIter = 50
nextStep = 0
batchSize = 64
dataFile = 'data'
labelFile = 'labels'


def saveNumpyData():
    trainData = []
    trainLabels = []

    for per in range(numPersons):

        personPath = os.path.join(root, persons[per], 'RGB')
        cameras = os.listdir(personPath)

        for cam in cameras:

            print('Loading', persons[per], cam)
            cameraPath = os.path.join(personPath, cam)
            files = os.listdir(cameraPath)
            framesCount = len(files)
            frames = np.ndarray([framesCount, rows, cols, channels])

            for i,file in enumerate(files):

                seqImg = cv2.imread(os.path.join(cameraPath, file))
                seqImg = cv2.resize(seqImg, (rows, cols))
                frames[i,:,:,:] = seqImg

            maxpooledFrame = np.max(frames, 0) / 255
            trainData.append(maxpooledFrame)
            label = np.zeros([numPersons])
            label[per] = 1
            trainLabels.append(label)

    trainData = np.array(trainData)
    trainLabels = np.array(trainLabels)
    permutation = np.random.permutation(trainData.shape[0])
    trainData = trainData[permutation, :, :, :]
    trainLabels = trainLabels[permutation, :]
    np.save(dataFile, trainData)
    np.save(labelFile, trainLabels)


def loadNumpyDataGenerateBatches():

    trainData = np.load(dataFile+'.npy')
    trainLabels = np.load(labelFile+'.npy')

    dataBatches = []
    labelBatches = []

    i = 0
    while i < trainData.shape[0]:

        dataBatches.append(trainData[i:i+batchSize,:,:,:])
        labelBatches.append(trainLabels[i:i+batchSize,:])
        i = i + batchSize

    i = i - batchSize
    if i < trainData.shape[0]:

        dataBatches.append(trainData[i:,:,:,:])
        labelBatches.append(trainLabels[i:,:])

    return dataBatches, labelBatches

#saveNumpyData()

data,label = loadNumpyDataGenerateBatches()
img = cv2.imread(os.path.join(root,'0001','RGB','Camera 1', '0001C1T0001F001.jpg'))
print(img.shape)
plt.imshow(img)
plt.figure()
plt.imshow(cv2.resize(img, (64,64)))
plt.show()

def train():

    dataBatches, labelBatches = loadNumpyDataGenerateBatches()
    trainData = np.array(dataBatches)
    trainLabels = np.array(labelBatches)

    with tf.Session(graph=graph) as session:

        tf.global_variables_initializer().run()

        if nextStep > 0:

            saver.restore(session, 'models/model-'+str(nextStep))

        for i in range(nextStep, numIter):

            for X_batch, Y_batch in zip(dataBatches, labelBatches):

                pred, l, _ = session.run([tf_prediction, loss, optimizer], feed_dict={tf_train:X_batch, tf_labels:Y_batch})

                accu = sum(pred==np.argmax(Y_batch, 1)) / Y_batch.shape[0]
                print('Iteration',i)
                print('\tBatch Loss:',l)
                print('\tBatch Accuracy:',accu)


            saver.save(session, 'models/model', global_step=i)
            logfile = open('models/log.txt', 'a+')
            logfile.write('Iteration ' + str(i) + '\n')
            logfile.close()

        trainPred = session.run([tf_prediction], feed_dict={tf_train: trainData, tf_labels: trainLabels})
        trainAccu = sum(trainPred == np.argmax(trainLabels, 1)) / trainLabels.shape[0]
        print('Train Accuracy:', trainAccu)

#train()
