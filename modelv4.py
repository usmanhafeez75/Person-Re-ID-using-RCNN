import tensorflow as tf
import tensorflow.contrib.layers as lays
import tables
import os
import numpy as np

saveFile = os.path.join('D:\\','Python','Person Re ID Dataset','bbox_train', 'trainV2.h5')
hdf5_file = tables.open_file(saveFile, mode='r')
persons = list(hdf5_file.root._v_children)
numPersons = len(persons)

noFrames = 500
rows = 64
cols = 64
channels = 3

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

convResult = 512
fc1Units = 256
fc2Units = 128
keep_prob = 0.7

margin = 0.5
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

        single_feature_vector = tf.reduce_mean(fc2, 0)

        return single_feature_vector


    tf_anchor = tf.placeholder(tf.float32, [None,rows,cols,channels])
    tf_pos = tf.placeholder(tf.float32, [None,rows,cols,channels])
    tf_neg = tf.placeholder(tf.float32, [None,rows,cols,channels])

    tf_keep_prob = tf.placeholder_with_default(1.0, ())

    anchor_vec = getFeatureVector(tf_anchor)
    pos_vec = getFeatureVector(tf_pos)
    neg_vec = getFeatureVector(tf_neg)

    pos_dist = tf.reduce_sum(tf.square(anchor_vec - pos_vec))
    neg_dist = tf.reduce_sum(tf.square(anchor_vec - neg_vec))
    loss = tf.maximum(pos_dist - neg_dist + margin, 0)  + (tf.maximum(pos_dist, margin) - tf.minimum(neg_dist, margin))

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    saver = tf.train.Saver()


savedPerson = 'p0001'
nextPersonInd = 0# persons.index(savedPerson)
nextCamera = 0
numIter = 50
nextStep = 0


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

                    anchor = np.array(hdf5_file.root.__getattr__(persons[per]).__getattr__(cameras[cam]).__getattr__('frames')[:noFrames,:,:,:])

                    camIndices = list(set(range(len(cameras))) - set([cam]))
                    if len(camIndices) > 0:
                        pos_cam = np.random.choice(camIndices)
                    else:
                        pos_cam = cam

                    pos = np.array(hdf5_file.root.__getattr__(persons[per]).__getattr__(cameras[pos_cam]).__getattr__('frames')[:noFrames,:,:,:])

                    perIndices = list(set(range(numPersons)) - set([per]))
                    negPerson = np.random.choice(perIndices)
                    negCameras = list(hdf5_file.root.__getattr__(persons[negPerson])._v_children)
                    neg_cam = np.random.randint(len(negCameras))

                    neg = np.array(hdf5_file.root.__getattr__(persons[negPerson]).__getattr__(negCameras[neg_cam]).__getattr__('frames')[:noFrames,:,:,:])

                    print('\tpos ', persons[per], cameras[pos_cam])
                    print('\tneg ', persons[negPerson], negCameras[neg_cam])

                    posDist, negDist, l, _ = session.run([pos_dist, neg_dist, loss, optimizer],
                                                       feed_dict = {tf_neg:neg, tf_anchor:anchor, tf_pos:pos,tf_keep_prob:keep_prob})

                    print('\tpos dist:', posDist)
                    print('\tneg dist:', negDist)
                    print('\tLoss:',l)

            saver.save(session, 'models/model', global_step=i)
            logfile = open('models/log.txt', 'a+')
            logfile.write('Iteration ' + str(i) + '\n')
            logfile.close()


def trainAccuracy(step):

    correct = 0
    total = 0
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        saver.restore(session, 'models/model-' +str(step))

        for per in range(numPersons):

            cameras = list(hdf5_file.root.__getattr__(persons[per])._v_children)

            for cam in range(len(cameras)):

                print('Evaluating '+  persons[per] + ' ' + cameras[cam])

                anchor = np.array \
                    (hdf5_file.root.__getattr__(persons[per]).__getattr__(cameras[cam]).__getattr__('frames')[:noFrames,:,:,:])

                camIndices = list(set(range(len(cameras))) - set([cam]))
                if len(camIndices) > 0:
                    pos_cam = np.random.choice(camIndices)
                else:
                    pos_cam = cam

                pos = np.array \
                    (hdf5_file.root.__getattr__(persons[per]).__getattr__(cameras[pos_cam]).__getattr__('frames')[:noFrames,:,:,:])

                perIndices = list(set(range(numPersons)) - set([per]))
                negPerson = np.random.choice(perIndices)
                negCameras = list(hdf5_file.root.__getattr__(persons[negPerson])._v_children)
                neg_cam = np.random.randint(len(negCameras))

                neg = np.array \
                    (hdf5_file.root.__getattr__(persons[negPerson]).__getattr__(negCameras[neg_cam]).__getattr__
                        ('frames')[:noFrames,:,:,:])

                positive, negative = session.run([pos_dist, neg_dist], feed_dict = {tf_neg :neg, tf_anchor :anchor, tf_pos :pos})
                total = total +1
                print('\tpos ', persons[per], cameras[pos_cam], positive)
                print('\tneg ', persons[negPerson], negCameras[neg_cam], negative)
                if negative - positive >= margin:

                    correct = correct + 1

    return correct / total,  total, correct


def testAccuracy(step):

    testFile = os.path.join('D:\\','Python','Person Re ID Dataset','bbox_test', 'test.h5')
    hdf5_fileTest = tables.open_file(testFile, mode='r')
    testPersons = list(hdf5_fileTest.root._v_children)
    numPersonsTest = len(testPersons)

    correct = 0
    total = 0
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        saver.restore(session, 'models/model-' +str(step))

        for per in range(numPersonsTest):

            cameras = list(hdf5_fileTest.root.__getattr__(testPersons[per])._v_children)

            for cam in range(len(cameras)):

                print('Evaluating '+  testPersons[per] + ' ' + cameras[cam])

                anchor = np.array \
                    (hdf5_fileTest.root.__getattr__(testPersons[per]).__getattr__(cameras[cam]).__getattr__('frames')[:noFrames,:,:,:])

                camIndices = list(set(range(len(cameras))) - set([cam]))
                if len(camIndices) > 0:
                    pos_cam = np.random.choice(camIndices)
                else:
                    pos_cam = cam

                pos = np.array \
                    (hdf5_fileTest.root.__getattr__(testPersons[per]).__getattr__(cameras[pos_cam]).__getattr__('frames')[:noFrames,:,:,:])

                perIndices = list(set(range(numPersons)) - set([per]))
                negPerson = np.random.choice(perIndices)
                negCameras = list(hdf5_fileTest.root.__getattr__(testPersons[negPerson])._v_children)
                neg_cam = np.random.randint(len(negCameras))

                neg = np.array \
                    (hdf5_fileTest.root.__getattr__(testPersons[negPerson]).__getattr__(negCameras[neg_cam]).__getattr__
                        ('frames'[:noFrames,:,:,:]))

                positive, negative = session.run([pos_dist, neg_dist], feed_dict = {tf_neg :neg, tf_anchor :anchor, tf_pos :pos})
                total = total +1
                print('\tpos ', testPersons[per], cameras[pos_cam], positive)
                print('\tneg ', testPersons[negPerson], negCameras[neg_cam], negative)
                if negative - positive >= margin:

                    correct = correct + 1

    hdf5_fileTest.close()
    return correct / total, total, correct


train()

print(trainAccuracy(numIter))

#print(testAccuracy(numIter))



hdf5_file.close()
