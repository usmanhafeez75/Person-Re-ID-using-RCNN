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
persons = list(hdf5_file.root._v_children)
numPersons = len(persons)

noFrames = 200
rows = 64
cols = 64
channels = 5

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

    def RNN(features):

        features = tf.unstack(features, noFrames, 0)

        rnn_cell = rnn.BasicRNNCell(embeddingSize, activation=tf.nn.tanh, reuse=tf.AUTO_REUSE)

        outputs, states = rnn.static_rnn(rnn_cell, features, dtype=tf.float32)

        return outputs

    def getFeatureVector(inputs):

        conv = CNN(inputs)
        conv = lays.flatten(conv)

        fc1 = tf.nn.tanh(tf.matmul(conv, Wf1) + bf1)
        fc2 = tf.nn.tanh(tf.matmul(fc1, Wf2) + bf2)

        fc2 = tf.reshape(fc2, [noFrames, 1, fc2Units])
        feature_vectors = RNN(fc2)
        single_feature_vector = tf.reduce_mean(feature_vectors, axis=0)

        return single_feature_vector


    tf_anchor = tf.placeholder(tf.float32, [noFrames,rows,cols,channels])
    tf_pos = tf.placeholder(tf.float32, [noFrames,rows,cols,channels])
    tf_neg = tf.placeholder(tf.float32, [noFrames,rows,cols,channels])
    tf_labels = tf.placeholder(tf.float32, [3, numPersons])


    tf_keep_prob = tf.placeholder_with_default(1.0, ())
    tf_keep_prob_rnn = tf.placeholder_with_default(1.0, ())

    anchor_vec = getFeatureVector(tf_anchor)
    pos_vec = getFeatureVector(tf_pos)
    neg_vec = getFeatureVector(tf_neg)

    anchor_logits = tf.matmul(anchor_vec, Wf3) + bf3
    pos_logits = tf.matmul(pos_vec, Wf3) + bf3
    neg_logits = tf.matmul(neg_vec, Wf3) + bf3

    logits = tf.reshape(tf.stack([anchor_logits, pos_logits, neg_logits], 0), [3,numPersons])

    logits_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_labels))

    tf_prediction = tf.argmax(tf.nn.softmax(logits), 1)

    pos_dist = tf.reduce_sum(tf.square(anchor_vec - pos_vec))
    neg_dist = tf.reduce_sum(tf.square(anchor_vec - neg_vec))
    triplet_loss = (tf.maximum(pos_dist, margin) - tf.minimum(neg_dist, margin)) + tf.maximum(pos_dist - neg_dist + margin, 0)

    loss = triplet_loss + logits_loss
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    saver = tf.train.Saver()


persons = list(hdf5_file.root._v_children)
numPersons = len(persons)
savedPerson = 'p0001'
nextPersonInd = 0# persons.index(savedPerson)
nextCamera = 0
numIter = 50
nextStep = 0

def reduceFrames(frames):

    if frames.shape[0] < noFrames:
        frames = np.append(frames, np.zeros([noFrames - frames.shape[0], rows, cols, channels], np.float32), 0)
    frames = np.transpose(np.reshape(frames, [-1, rows * cols * channels]))
    frames = np.nan_to_num(frames)
    scaler = StandardScaler()
    pca = PCA(n_components=noFrames)
    scaler.fit(frames)
    frames = scaler.transform(frames)
    frames = np.nan_to_num(frames)
    pca.fit(frames)
    frames = pca.transform(frames)
    frames = np.reshape(np.transpose(frames), [noFrames, rows, cols, channels])

    return frames


def train():

    with tf.Session(graph=graph) as session:

        tf.global_variables_initializer().run()

        if nextPersonInd > 0 or nextStep > 0:

            saver.restore(session, 'models/model-'+str(nextStep))

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

                    anchor = reduceFrames(anchor)
                    pos = reduceFrames(pos)
                    neg = reduceFrames(neg)

                    labels = np.zeros((3,numPersons))
                    labels[:2,per] = 1
                    labels[2,negPerson] = 1

                    posDist, negDist, ll, tl, l, opt, pred = session.run([pos_dist, neg_dist, logits_loss, triplet_loss, loss, optimizer, tf_prediction],
                                                       feed_dict = {tf_neg:neg, tf_anchor:anchor, tf_pos:pos,
                                                                         tf_keep_prob:keep_prob, tf_keep_prob_rnn:keep_prob_rnn,
                                                                         tf_labels:labels})

                    print('\t',persons[pred[0]], persons[pred[1]], persons[pred[2]])
                    print('\tpos dist:', posDist)
                    print('\tneg dist:', negDist)
                    print('\tLogits Loss:',ll)
                    print('\tTriplet Loss:', tl)
                    print('\tTotal Loss:',l)

            saver.save(session, 'models/model', global_step=i)
            logfile = open('models/log.txt', 'a+')
            logfile.write('Iteration ' + str(i) + '\n')
            logfile.close()
        else:
            saver.restore(session, 'models/model-'+str(nextStep))


def trainAccuracy(step):

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

                anchor = reduceFrames(anchor)
                pos = reduceFrames(pos)
                neg = reduceFrames(neg)

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

    return correctPos / total, correctNeg/total, total, correctPos, correctNeg, posOdds, negOdds


def testAccuracy(step):

    testFile = os.path.join('D:\\','Python','Person Re ID Dataset','bbox_test', 'test.h5')
    hdf5_fileTest = tables.open_file(testFile, mode='r')
    testPersons = list(hdf5_fileTest.root._v_children)
    numPersonsTest = len(testPersons)

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

                anchor = reduceFrames(anchor)
                pos = reduceFrames(pos)
                neg = reduceFrames(neg)

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

    hdf5_fileTest.close()
    return correctPos / total, correctNeg/total, total, correctPos, correctNeg, posOdds, negOdds

#
# with tf.Session(graph=graph)as session:
#     tf.global_variables_initializer().run()
#     anchor = reduceFrames(np.array(hdf5_file.root.__getattr__(persons[0]).__getattr__('c1').__getattr__('frames')))
#     a,b = session.run([anchor_vec, pos_vec], feed_dict={tf_anchor:anchor, tf_pos:anchor})
#     print(a)
#     print('----------------------------')
#     print(b)
#     print(a==b)


train()

print(trainAccuracy(numIter))

print(testAccuracy(numIter))



hdf5_file.close()
