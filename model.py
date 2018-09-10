import tensorflow as tf
import tensorflow.contrib.layers as lays
import tensorflow.contrib.rnn as rnn
import tables
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

saveFile = os.path.join('D:\\','Python','Person Re ID Dataset','bbox_train', 'train.h5')
hdf5_file = tables.open_file(saveFile, mode='r')

noFrames = 5
rows = 64
cols = 64
channels = 5

filter1 = 16
filter2 = 32
filter3 = 32
filterSize = [5,5]
stride = 1
padding = 'SAME'
maxpoolSize = [2,2]
poolStride = 2
poolPadding = 'VALID'

fc1Units = 1024
fc2Units = 512

embeddingSize = 128
keep_prob = 0.7
keep_prob_rnn = 0.7

margin = 100
learning_rate = 0.01

isTrain = True

graph = tf.Graph()
with graph.as_default():

    def CNN(frames):


        conv1 = lays.conv2d(frames, filter1, filterSize, stride=stride, padding=padding, activation_fn=tf.nn.tanh)
        pool1 = lays.max_pool2d(conv1, maxpoolSize, stride=poolStride, padding=poolPadding)


        conv2 = lays.conv2d(pool1, filter2, filterSize, stride=stride, padding=padding, activation_fn=tf.nn.tanh)
        pool2 = lays.max_pool2d(conv2, maxpoolSize, stride=poolStride, padding=poolPadding)

        conv3 = lays.conv2d(pool2, filter3, filterSize, stride=stride, padding=padding, activation_fn=tf.nn.tanh)
        pool3 = lays.max_pool2d(conv3, maxpoolSize, stride=poolStride, padding=poolPadding)

        return pool3

    def RNN(features):

        features = tf.unstack(features, noFrames, 0)

        rnn_cell = rnn.BasicRNNCell(embeddingSize, activation=tf.nn.tanh, reuse=tf.AUTO_REUSE)

        outputs, states = rnn.static_rnn(rnn_cell, features, dtype=tf.float32)

        return outputs

    def getFeatureVector(inputs):

        conv = CNN(inputs)
        conv_flatten = lays.flatten(conv)

        fc1 = lays.fully_connected(conv_flatten, fc1Units, tf.nn.tanh)
        if isTrain:
            fc1 = lays.dropout(fc1, keep_prob)
        else:
            fc1 = lays.dropout(fc1, 1)

        fc2 = lays.fully_connected(fc1, fc2Units, tf.nn.tanh)
        if isTrain:
            fc2 = lays.dropout(fc2, keep_prob_rnn)
        else:
            fc2 = lays.dropout(fc2, 1)

        fc2 = tf.reshape(fc2, [noFrames, 1, fc2Units])
        feature_vectors = RNN(fc2)
        single_feature_vector = tf.reduce_mean(feature_vectors, axis=0)

        return single_feature_vector


    tf_anchor = tf.placeholder(tf.float32, [noFrames,rows,cols,channels])
    tf_pos = tf.placeholder(tf.float32, [noFrames,rows,cols,channels])
    tf_neg = tf.placeholder(tf.float32, [noFrames,rows,cols,channels])

    anchor_vec = getFeatureVector(tf_anchor)
    pos_vec = getFeatureVector(tf_pos)
    neg_vec = getFeatureVector(tf_neg)

    pos_dist = tf.reduce_sum(tf.square(anchor_vec - pos_vec))
    neg_dist = tf.reduce_sum(tf.square(anchor_vec - neg_vec))
    loss = (tf.maximum(pos_dist, margin) - tf.minimum(neg_dist, margin)) + tf.maximum(pos_dist - neg_dist + margin, 0)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    saver = tf.train.Saver()


persons = list(hdf5_file.root._v_children)
numPersons = len(persons)
savedPerson = 'p0001'
nextPersonInd = 0# persons.index(savedPerson)
nextCamera = 0
numIter = 4
nextStep = 0

def train():

    global isTrain
    isTrain = True

    with tf.Session(graph=graph) as session:

        tf.global_variables_initializer().run()

        if nextPersonInd > 0 or nextStep > 0:

            saver.restore(session, 'models/model-'+str(nextStep))

        if isTrain:

            for i in range(nextStep,numIter):

                for per in range(nextPersonInd, numPersons):

                    cameras = list(hdf5_file.root.__getattr__(persons[per])._v_children)

                    for cam in range(nextCamera, len(cameras)):

                        print('Iteration', i, 'Processing '+  persons[per] + ' ' + cameras[cam])

                        anchor = np.array(hdf5_file.root.__getattr__(persons[per]).__getattr__(cameras[cam]).__getattr__('frames'))

                        camIndices = list(set(range(len(cameras))) - set([cam]))
                        if len(camIndices) > 0:
                            pos_cam = np.random.choice(camIndices)
                        else:
                            pos_cam = cam

                        pos = np.array(hdf5_file.root.__getattr__(persons[per]).__getattr__(cameras[pos_cam]).__getattr__('frames'))

                        perIndices = list(set(range(numPersons)) - set([per]))
                        negPerson = np.random.choice(perIndices)
                        negCameras = list(hdf5_file.root.__getattr__(persons[negPerson])._v_children)
                        neg_cam = np.random.randint(len(negCameras))

                        neg = np.array(hdf5_file.root.__getattr__(persons[negPerson]).__getattr__(negCameras[neg_cam]).__getattr__('frames'))

                        print('\tpos ', persons[per], cameras[pos_cam])
                        print('\tneg ', persons[negPerson], negCameras[neg_cam])

                        anchor = np.transpose(np.reshape(anchor, [-1, rows * cols * channels]))
                        pos = np.transpose(np.reshape(pos, [-1, rows * cols * channels]))
                        neg = np.transpose(np.reshape(neg, [-1, rows * cols * channels]))


                        anchor = np.nan_to_num(anchor)
                        scaler = StandardScaler()
                        pca = PCA(n_components=noFrames)
                        scaler.fit(anchor)
                        anchor = scaler.transform(anchor)
                        anchor = np.nan_to_num(anchor)
                        pca.fit(anchor)
                        anchor = pca.transform(anchor)

                        pos = np.nan_to_num(pos)
                        scaler = StandardScaler()
                        pca = PCA(n_components=noFrames)
                        scaler.fit(pos)
                        pos = scaler.transform(pos)
                        pos = np.nan_to_num(pos)
                        pca.fit(pos)
                        pos = pca.transform(pos)

                        neg = np.nan_to_num(neg)
                        scaler = StandardScaler()
                        pca = PCA(n_components=noFrames)
                        scaler.fit(neg)
                        neg = scaler.transform(neg)
                        neg = np.nan_to_num(neg)
                        pca.fit(neg)
                        neg = pca.transform(neg)

                        anchor = np.reshape(np.transpose(anchor), [noFrames, rows, cols, channels])
                        pos = np.reshape(np.transpose(pos), [noFrames, rows, cols, channels])
                        neg = np.reshape(np.transpose(neg), [noFrames, rows, cols, channels])

                        l, opt = session.run([loss, optimizer], feed_dict = {tf_neg:neg, tf_anchor:anchor, tf_pos:pos})
                        print('\tLoss:',l)

                saver.save(session, 'models/model', global_step=i)
                logfile = open('models/log.txt', 'a+')
                logfile.write('Iteration ' + str(i) + '\n')
                logfile.close()
        else:
            saver.restore(session, 'models/model-'+str(nextStep))

    isTrain = False


def trainAccuracy(step):

    global isTrain
    isTrain= False
    correctNeg = 0
    total = 0
    correctPos = 0
    posOdds = 0
    negOdds = 0
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        saver.restore(session, 'models/model-' +str(step))

        for per in range(numPersons):

            cameras = list(hdf5_file.root.__getattr__(persons[per])._v_children)

            for cam in range(len(cameras)):

                print('Evaluating '+  persons[per] + ' ' + cameras[cam])

                anchor = np.array \
                    (hdf5_file.root.__getattr__(persons[per]).__getattr__(cameras[cam]).__getattr__('frames'))

                camIndices = list(set(range(len(cameras))) - set([cam]))
                if len(camIndices) > 0:
                    pos_cam = np.random.choice(camIndices)
                else:
                    pos_cam = cam

                pos = np.array \
                    (hdf5_file.root.__getattr__(persons[per]).__getattr__(cameras[pos_cam]).__getattr__('frames'))

                perIndices = list(set(range(numPersons)) - set([per]))
                negPerson = np.random.choice(perIndices)
                negCameras = list(hdf5_file.root.__getattr__(persons[negPerson])._v_children)
                neg_cam = np.random.randint(len(negCameras))

                neg = np.array \
                    (hdf5_file.root.__getattr__(persons[negPerson]).__getattr__(negCameras[neg_cam]).__getattr__
                        ('frames'))

                anchor = np.transpose(np.reshape(anchor, [-1, rows * cols * channels]))
                pos = np.transpose(np.reshape(pos, [-1, rows * cols * channels]))
                neg = np.transpose(np.reshape(neg, [-1, rows * cols * channels]))


                anchor = np.nan_to_num(anchor)
                scaler = StandardScaler()
                pca = PCA(n_components=noFrames)
                scaler.fit(anchor)
                anchor = scaler.transform(anchor)
                anchor = np.nan_to_num(anchor)
                pca.fit(anchor)
                anchor = pca.transform(anchor)

                pos = np.nan_to_num(pos)
                scaler = StandardScaler()
                pca = PCA(n_components=noFrames)
                scaler.fit(pos)
                pos = scaler.transform(pos)
                pos = np.nan_to_num(pos)
                pca.fit(pos)
                pos = pca.transform(pos)

                neg = np.nan_to_num(neg)
                scaler = StandardScaler()
                pca = PCA(n_components=noFrames)
                scaler.fit(neg)
                neg = scaler.transform(neg)
                neg = np.nan_to_num(neg)
                pca.fit(neg)
                neg = pca.transform(neg)

                anchor = np.reshape(np.transpose(anchor), [noFrames, rows, cols, channels])
                pos = np.reshape(np.transpose(pos), [noFrames, rows, cols, channels])
                neg = np.reshape(np.transpose(neg), [noFrames, rows, cols, channels])

                positive, negative = session.run([pos_dist, neg_dist], feed_dict = {tf_neg :neg, tf_anchor :anchor, tf_pos :pos})
                total = total +1
                print('\tpos ', persons[per], cameras[pos_cam], positive)
                print('\tneg ', persons[negPerson], negCameras[neg_cam], negative)
                if positive < margin:
                    correctPos = correctPos + 1
                if negative > margin:
                    correctNeg = correctNeg + 1
                if positive == margin:
                    posOdds = posOdds + 1
                if negative == margin:
                    negOdds = negOdds + 1

    isTrain = True
    return correctPos / total, correctNeg/total, total, correctPos, correctNeg, posOdds, negOdds


def testAccuracy(step):

    testFile = os.path.join('D:\\','Python','Person Re ID Dataset','bbox_test', 'test.h5')
    hdf5_fileTest = tables.open_file(testFile, mode='r')
    testPersons = list(hdf5_fileTest.root._v_children)
    numPersonsTest = len(testPersons)

    global isTrain
    isTrain= False
    correctNeg = 0
    total = 0
    correctPos = 0
    posOdds = 0
    negOdds = 0
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        saver.restore(session, 'models/model-' +str(step))

        for per in range(numPersonsTest):

            cameras = list(hdf5_fileTest.root.__getattr__(testPersons[per])._v_children)

            for cam in range(len(cameras)):

                print('Evaluating '+  testPersons[per] + ' ' + cameras[cam])

                anchor = np.array \
                    (hdf5_fileTest.root.__getattr__(testPersons[per]).__getattr__(cameras[cam]).__getattr__('frames'))

                camIndices = list(set(range(len(cameras))) - set([cam]))
                if len(camIndices) > 0:
                    pos_cam = np.random.choice(camIndices)
                else:
                    pos_cam = cam

                pos = np.array \
                    (hdf5_fileTest.root.__getattr__(testPersons[per]).__getattr__(cameras[pos_cam]).__getattr__('frames'))

                perIndices = list(set(range(numPersons)) - set([per]))
                negPerson = np.random.choice(perIndices)
                negCameras = list(hdf5_fileTest.root.__getattr__(testPersons[negPerson])._v_children)
                neg_cam = np.random.randint(len(negCameras))

                neg = np.array \
                    (hdf5_fileTest.root.__getattr__(testPersons[negPerson]).__getattr__(negCameras[neg_cam]).__getattr__
                        ('frames'))

                if anchor.shape[0] < noFrames:
                    anchor = np.append(anchor, np.zeros([noFrames - anchor.shape[0], rows, cols, channels], np.float32), 0)

                if pos.shape[0] < noFrames:
                    pos = np.append(pos, np.zeros([noFrames - pos.shape[0], rows, cols, channels], np.float32), 0)

                if neg.shape[0] < noFrames:
                    neg = np.append(neg, np.zeros([noFrames - neg.shape[0], rows, cols, channels], np.float32), 0)

                anchor = np.transpose(np.reshape(anchor, [-1, rows * cols * channels]))
                pos = np.transpose(np.reshape(pos, [-1, rows * cols * channels]))
                neg = np.transpose(np.reshape(neg, [-1, rows * cols * channels]))


                anchor = np.nan_to_num(anchor)
                scaler = StandardScaler()
                pca = PCA(n_components=noFrames)
                scaler.fit(anchor)
                anchor = scaler.transform(anchor)
                anchor = np.nan_to_num(anchor)
                pca.fit(anchor)
                anchor = pca.transform(anchor)

                pos = np.nan_to_num(pos)
                scaler = StandardScaler()
                pca = PCA(n_components=noFrames)
                scaler.fit(pos)
                pos = scaler.transform(pos)
                pos = np.nan_to_num(pos)
                pca.fit(pos)
                pos = pca.transform(pos)

                neg = np.nan_to_num(neg)
                scaler = StandardScaler()
                pca = PCA(n_components=noFrames)
                scaler.fit(neg)
                neg = scaler.transform(neg)
                neg = np.nan_to_num(neg)
                pca.fit(neg)
                neg = pca.transform(neg)

                anchor = np.reshape(np.transpose(anchor), [noFrames, rows, cols, channels])
                pos = np.reshape(np.transpose(pos), [noFrames, rows, cols, channels])
                neg = np.reshape(np.transpose(neg), [noFrames, rows, cols, channels])

                positive, negative = session.run([pos_dist, neg_dist], feed_dict = {tf_neg :neg, tf_anchor :anchor, tf_pos :pos})
                total = total +1
                print('\tpos ', testPersons[per], cameras[pos_cam], positive)
                print('\tneg ', testPersons[negPerson], negCameras[neg_cam], negative)
                if positive < margin:
                    correctPos = correctPos + 1
                if negative > margin:
                    correctNeg = correctNeg + 1
                if positive == margin:
                    posOdds = posOdds + 1
                if negative == margin:
                    negOdds = negOdds + 1

    isTrain = True
    hdf5_fileTest.close()
    return correctPos / total, correctNeg/total, total, correctPos, correctNeg, posOdds, negOdds


train()

print(trainAccuracy(numIter))

print(testAccuracy(1))



hdf5_file.close()
