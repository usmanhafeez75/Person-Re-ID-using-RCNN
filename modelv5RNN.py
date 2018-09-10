import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tables
import os
import numpy as np
import tensorflow.contrib.layers as lays
import time


trainFeaturesFile = os.path.join('D:\\','Python','Person Re ID Dataset','bbox_train', 'trainV3CNNFeatures.h5')
os.path.join('D:\\','Python','Person Re ID Dataset','bbox_test', 'testV3CNNFeatures.h5')
train_hdf5_file = tables.open_file(trainFeaturesFile, mode='r')

maxFrames = 128
featureVecSize = 625
embeddingSize = 128
keep_prob = 0.9
margin = 0.5
maxGrad = 100.0

graph = tf.Graph()
with graph.as_default():

    tf_anchor = tf.placeholder(tf.float32, [maxFrames,featureVecSize])
    tf_pos = tf.placeholder(tf.float32, [maxFrames,featureVecSize])
    tf_neg = tf.placeholder(tf.float32, [maxFrames,featureVecSize])
    tf_keep_prob = tf.placeholder_with_default(1.0, ())

    cell = rnn.BasicLSTMCell(embeddingSize, activation=tf.nn.tanh)
    W = tf.get_variable('W', [embeddingSize,embeddingSize], tf.float32, lays.xavier_initializer())
    b = tf.Variable(tf.zeros([embeddingSize]), dtype=tf.float32)

    def getEmbedding(features):

        features = tf.reshape(features, [maxFrames,1,featureVecSize])
        features = tf.unstack(features, maxFrames, 0)
        outputs, _ = rnn.static_rnn(cell, features, dtype=tf.float32)
        outputs = outputs[-1]#tf.reshape(outputs, [maxFrames,embeddingSize])
        #outputs = tf.contrib.layers.dropout(outputs, tf_keep_prob)
        outputs = tf.nn.tanh(tf.matmul(outputs,W) + b)
        return outputs


    tf_anchor_vec = getEmbedding(tf_anchor)
    tf_pos_vec = getEmbedding(tf_pos)
    tf_neg_vec = getEmbedding(tf_neg)

    tf_posDist = tf.reduce_sum(tf.square(tf_anchor_vec - tf_pos_vec))
    tf_negDist = tf.reduce_sum(tf.square(tf_anchor_vec - tf_neg_vec))

    loss = tf.maximum(tf_posDist, margin) - tf.minimum(tf_negDist, margin) #+ tf.maximum(tf_posDist - tf_negDist + margin/10, 0)
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    # grads = optimizer1.compute_gradients(loss)
    # grads_clipped = [(tf.clip_by_value(g,-maxGrad, maxGrad),v) for g,v in grads]
    # optimizer = optimizer1.apply_gradients(grads_clipped)
    saver = tf.train.Saver()


persons = list(train_hdf5_file.root._v_children)
numPersons = len(persons)
numIter = 500
nextStep = 154

def train():

    with tf.Session(graph=graph) as session:

        tf.global_variables_initializer().run()

        if nextStep > 0:

            saver.restore(session, 'modelsRNN/model-'+str(nextStep - 1))

        for i in range(nextStep, numIter):

            for per in range(numPersons):

                cameras = list(train_hdf5_file.root.__getattr__(persons[per])._v_children)

                for cam in range(len(cameras)):

                    print('Iteration', i, 'Processing ' + persons[per] + ' ' + cameras[cam])

                    anchor = np.array(
                        train_hdf5_file.root.__getattr__(persons[per]).__getattr__(cameras[cam]).__getattr__('features')[:maxFrames,:])

                    camIndices = list(set(range(len(cameras))) - set([cam]))

                    if len(camIndices) == 0:
                        camIndices = [cam]

                    for pos_cam in camIndices:

                        pos = np.array(
                            train_hdf5_file.root.__getattr__(persons[per]).__getattr__(cameras[pos_cam]).__getattr__('features')[:maxFrames,:])

                        perIndices = list(set(range(numPersons)) - set([per]))
                        negPerson = np.random.choice(perIndices)
                        negCameras = list(train_hdf5_file.root.__getattr__(persons[negPerson])._v_children)
                        neg_cam = np.random.randint(len(negCameras))

                        neg = np.array(
                            train_hdf5_file.root.__getattr__(persons[negPerson]).__getattr__(negCameras[neg_cam]).__getattr__(
                                'features')[:maxFrames,:])

                        print('\tpos ', persons[per], cameras[pos_cam])
                        print('\tneg ', persons[negPerson], negCameras[neg_cam])

                        if anchor.shape[0] < maxFrames:
                            anchor = np.append(anchor,np.zeros([maxFrames - anchor.shape[0],featureVecSize], np.float32), 0)

                        if pos.shape[0] < maxFrames:
                            pos = np.append(pos, np.zeros([maxFrames - pos.shape[0],featureVecSize], np.float32), 0)

                        if neg.shape[0] < maxFrames:
                            neg = np.append(neg, np.zeros([maxFrames - neg.shape[0],featureVecSize], np.float32), 0)

                        posDist,negDist,l,_ = session.run([tf_posDist,tf_negDist,loss,optimizer], feed_dict={tf_anchor:anchor,
                                                                                                             tf_pos:pos,
                                                                                                             tf_neg:neg,
                                                                                                             tf_keep_prob:keep_prob})
                        print('\tPosDist:',posDist)
                        print('\tNegDist:',negDist)
                        print('\tLoss:',l,'\n')
                        time.sleep(1)


            saver.save(session, 'modelsRNN/model', global_step=i)


def trainAccuracy(step):

    total = 0
    posCorrect = 0
    negCorrect = 0
    correct = 0
    negMisc = 0
    posMisc = 0

    with tf.Session(graph=graph) as session:

        tf.global_variables_initializer().run()

        if step is not None:

            saver.restore(session, 'modelsRNN/model-'+str(step))

        for per in range(numPersons):

            cameras = list(train_hdf5_file.root.__getattr__(persons[per])._v_children)

            for cam in range(len(cameras)):

                print('Evaluating ' + persons[per] + ' ' + cameras[cam])

                anchor = np.array(
                    train_hdf5_file.root.__getattr__(persons[per]).__getattr__(cameras[cam]).__getattr__('features')[
                    :maxFrames, :])

                camIndices = list(set(range(len(cameras))) - set([cam]))
                if len(camIndices) == 0:
                    camIndices = [cam]

                for pos_cam in camIndices:

                    pos = np.array(
                        train_hdf5_file.root.__getattr__(persons[per]).__getattr__(cameras[pos_cam]).__getattr__(
                            'features')[:maxFrames, :])

                    perIndices = list(set(range(numPersons)) - set([per]))
                    negPerson = np.random.choice(perIndices)
                    negCameras = list(train_hdf5_file.root.__getattr__(persons[negPerson])._v_children)
                    neg_cam = np.random.randint(len(negCameras))

                    neg = np.array(
                        train_hdf5_file.root.__getattr__(persons[negPerson]).__getattr__(
                            negCameras[neg_cam]).__getattr__(
                            'features')[:maxFrames, :])

                    print('\tpos ', persons[per], cameras[pos_cam])
                    print('\tneg ', persons[negPerson], negCameras[neg_cam])

                    if anchor.shape[0] < maxFrames:
                        anchor = np.append(anchor, np.zeros([maxFrames - anchor.shape[0], featureVecSize], np.float32),
                                           0)

                    if pos.shape[0] < maxFrames:
                        pos = np.append(pos, np.zeros([maxFrames - pos.shape[0], featureVecSize], np.float32), 0)

                    if neg.shape[0] < maxFrames:
                        neg = np.append(neg, np.zeros([maxFrames - neg.shape[0], featureVecSize], np.float32), 0)

                    posDist, negDist, l= session.run([tf_posDist, tf_negDist, loss],
                                                         feed_dict={tf_anchor: anchor,
                                                                    tf_pos: pos,
                                                                    tf_neg: neg})
                    print('\tPosDist:', posDist)
                    print('\tNegDist:', negDist)
                    print('\tLoss:', l, '\n')

                    total = total + 1

                    if posDist < margin:
                        posCorrect = posCorrect + 1

                    if negDist > margin:
                        negCorrect = negCorrect + 1

                    if posDist < margin and negDist > margin:
                        correct = correct + 1

                    if posDist == margin:
                        posMisc = posMisc + 1

                    if negDist == margin:
                        negMisc = negMisc + 1

                    time.sleep(1)

    return posCorrect,negCorrect,correct,total,posCorrect/total,negCorrect/total,correct/total,posMisc,negMisc


def testAccuracy(step):

    testFeaturesFile = os.path.join('D:\\', 'Python', 'Person Re ID Dataset', 'bbox_test', 'testV3CNNFeatures.h5')
    test_hdf5_file = tables.open_file(testFeaturesFile, mode='r')
    testpersons = list(test_hdf5_file.root._v_children)
    testnumPersons = len(testpersons)

    total = 0
    posCorrect = 0
    negCorrect = 0
    correct = 0
    negMisc = 0
    posMisc = 0

    with tf.Session(graph=graph) as session:

        tf.global_variables_initializer().run()

        if step is not None:
            saver.restore(session, 'modelsRNN/model-' + str(step))

        for per in range(testnumPersons):

            cameras = list(test_hdf5_file.root.__getattr__(testpersons[per])._v_children)

            for cam in range(len(cameras)):

                print('Evaluating ' + testpersons[per] + ' ' + cameras[cam])

                anchor = np.array(
                    test_hdf5_file.root.__getattr__(testpersons[per]).__getattr__(cameras[cam]).__getattr__('features')[
                    :maxFrames, :])

                camIndices = list(set(range(len(cameras))) - set([cam]))
                if len(camIndices) == 0:
                    camIndices = [cam]

                for pos_cam in camIndices:

                    pos = np.array(
                        test_hdf5_file.root.__getattr__(testpersons[per]).__getattr__(cameras[pos_cam]).__getattr__(
                            'features')[:maxFrames, :])

                    perIndices = list(set(range(testnumPersons)) - set([per]))
                    negPerson = np.random.choice(perIndices)
                    negCameras = list(test_hdf5_file.root.__getattr__(testpersons[negPerson])._v_children)
                    neg_cam = np.random.randint(len(negCameras))

                    neg = np.array(
                        test_hdf5_file.root.__getattr__(testpersons[negPerson]).__getattr__(
                            negCameras[neg_cam]).__getattr__(
                            'features')[:maxFrames, :])

                    print('\tpos ', testpersons[per], cameras[pos_cam])
                    print('\tneg ', testpersons[negPerson], negCameras[neg_cam])

                    if anchor.shape[0] < maxFrames:
                        anchor = np.append(anchor, np.zeros([maxFrames - anchor.shape[0], featureVecSize], np.float32),
                                           0)

                    if pos.shape[0] < maxFrames:
                        pos = np.append(pos, np.zeros([maxFrames - pos.shape[0], featureVecSize], np.float32), 0)

                    if neg.shape[0] < maxFrames:
                        neg = np.append(neg, np.zeros([maxFrames - neg.shape[0], featureVecSize], np.float32), 0)

                    posDist, negDist, l = session.run([tf_posDist, tf_negDist, loss],
                                                      feed_dict={tf_anchor: anchor,
                                                                 tf_pos: pos,
                                                                 tf_neg: neg})
                    print('\tPosDist:', posDist)
                    print('\tNegDist:', negDist)
                    print('\tLoss:', l, '\n')

                    total = total + 1

                    if posDist < margin:
                        posCorrect = posCorrect + 1

                    if negDist > margin:
                        negCorrect = negCorrect + 1

                    if posDist < margin and negDist > margin:
                        correct = correct + 1

                    if posDist == margin:
                        posMisc = posMisc + 1

                    if negDist == margin:
                        negMisc = negMisc + 1

                    time.sleep(1)

    test_hdf5_file.close()

    return posCorrect, negCorrect, correct, total, posCorrect / total, negCorrect / total, correct / total, posMisc, negMisc


def viperAccuracy(step):

    cam_a_path = os.path.join('D:\\','Python','Person Re ID Dataset','VIPeR','cam_aCNN.h5')
    cam_b_path = os.path.join('D:\\','Python','Person Re ID Dataset','VIPeR','cam_bCNN.h5')

    cam_a_hdf5_file = tables.open_file(cam_a_path, mode='r')
    cam_b_hdf5_file = tables.open_file(cam_b_path, mode='r')

    cam_a_features = cam_a_hdf5_file.root.__getattr__('features')
    cam_b_features = cam_b_hdf5_file.root.__getattr__('features')
    VnumPersons = cam_a_features.shape[0]

    total = 0
    posCorrect = 0
    negCorrect = 0
    correct = 0
    negMisc = 0
    posMisc = 0

    with tf.Session(graph=graph) as session:

        tf.global_variables_initializer().run()
        saver.restore(session, 'modelsRNN/model-' + str(step))

        for per in range(VnumPersons):

            anchor = np.array(cam_a_features[per])
            pos = np.array(cam_b_features[per])
            negIndices = list(set(range(VnumPersons)) - set([per]))
            negIndex = np.random.choice(negIndices)
            if per % 2 == 0:
                neg = np.array(cam_a_features[negIndex])
            else:
                neg = np.array(cam_b_features[negIndex])

            anchor = np.reshape(anchor, [1,featureVecSize])
            pos = np.reshape(pos, [1,featureVecSize])
            neg = np.reshape(neg, [1,featureVecSize])

            if anchor.shape[0] < maxFrames:
                anchor = np.append(anchor, np.zeros([maxFrames - anchor.shape[0], featureVecSize], np.float32),
                                   0)

            if pos.shape[0] < maxFrames:
                pos = np.append(pos, np.zeros([maxFrames - pos.shape[0], featureVecSize], np.float32), 0)

            if neg.shape[0] < maxFrames:
                neg = np.append(neg, np.zeros([maxFrames - neg.shape[0], featureVecSize], np.float32), 0)



            posDist, negDist, l = session.run([tf_posDist, tf_negDist, loss],
                                              feed_dict={tf_anchor: anchor,
                                                         tf_pos: pos,
                                                         tf_neg: neg})
            print('\tPosDist:', posDist)
            print('\tNegDist:', negDist)
            print('\tLoss:', l, '\n')

            total = total + 1

            if posDist < margin:
                posCorrect = posCorrect + 1

            if negDist > margin:
                negCorrect = negCorrect + 1

            if posDist < margin and negDist > margin:
                correct = correct + 1

            if posDist == margin:
                posMisc = posMisc + 1

            if negDist == margin:
                negMisc = negMisc + 1

    cam_a_hdf5_file.close()
    cam_b_hdf5_file.close()

    return posCorrect, negCorrect, correct, total, posCorrect / total, negCorrect / total, correct / total, posMisc, negMisc


#train()

#print(trainAccuracy(337))

print(testAccuracy(337))

# print(viperAccuracy(143))


train_hdf5_file.close()