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
rootDir = os.path.join('D:\\','Python','Person Re ID Dataset','bbox_train','bbox_train')
# saveFile = os.path.join('D:\\','Python','Person Re ID Dataset','bbox_train', 'trainV3.h5')


# hdf5_file = tables.open_file(saveFile, mode='r')
persons = os.listdir(rootDir)
numPersons = len(persons)

# frames_array = hdf5_file.root.__getattr__('shuffled_frames')
# labels_array = hdf5_file.root.__getattr__('shuffled_labels')
# total_train_size = frames_array.shape[0]

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
keep_prob = 0.7


isTrain = True

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


savedPerson = 'p0001'
nextPersonInd = 0# persons.index(savedPerson)
nextCamera = 0
numIter = 5
nextStep = 0
batch_size = 1024


def train():

    with tf.Session(graph=graph) as session:

        tf.global_variables_initializer().run()

        if nextPersonInd > 0 or nextStep > 0:

            saver.restore(session, 'models/model-'+str(nextStep))

        for i in range(nextStep,numIter):

            print('Iteration',i)
            start = 0
            batchNo = 0

            while start < total_train_size - batch_size:

                train_batch_X = frames_array[start:start+batch_size,:,:,:]
                train_batch_Y = labels_array[start:start+batch_size,:]
                l,_ = session.run([loss,optimizer], feed_dict={tf_train_X:train_batch_X,tf_train_Y:train_batch_Y,tf_keep_prob:keep_prob})
                print('\tBatch',batchNo,'Loss:',l)
                start = start + batch_size
                batchNo = batchNo + 1

            start = start - batch_size
            if start < total_train_size:

                train_batch_X = frames_array[start:, :, :, :]
                train_batch_Y = labels_array[start:, :]
                l, _ = session.run([loss, optimizer], feed_dict={tf_train_X: train_batch_X, tf_train_Y: train_batch_Y,tf_keep_prob: keep_prob})
                print('\tBatch', batchNo, 'Loss:', l)

            saver.save(session, 'models/model', global_step=i)
            logfile = open('models/log.txt', 'a+')
            logfile.write('Iteration ' + str(i) + '\n')
            logfile.close()


#   0.9846
def trainAccuracy(step):

    correct = 0
    total = frames_array.shape[0]

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        saver.restore(session, 'models/model-' +str(step))

        start = 0
        batchNo = 0

        while start < total_train_size - batch_size:

            train_batch_X = frames_array[start:start + batch_size, :, :, :]
            train_batch_Y = labels_array[start:start + batch_size, :]
            l,prediction = session.run([loss,tf_prediction], feed_dict={tf_train_X: train_batch_X, tf_train_Y: train_batch_Y})
            correctinBatch = np.sum(np.argmax(train_batch_Y,1) == np.argmax(prediction,1))
            print('\tBatch', batchNo, 'Correct Predictions:', correctinBatch, 'out of', train_batch_Y.shape[0])
            correct = correct + correctinBatch
            start = start + batch_size
            batchNo = batchNo + 1

        start = start - batch_size

        if start < total_train_size:

            train_batch_X = frames_array[start:, :, :, :]
            train_batch_Y = labels_array[start:, :]
            l,prediction = session.run([loss,tf_prediction], feed_dict={tf_train_X: train_batch_X, tf_train_Y: train_batch_Y})
            correctinBatch = np.sum(np.argmax(train_batch_Y, 1) == np.argmax(prediction, 1))
            print('\tBatch', batchNo, 'Correct Predictions:', correctinBatch, 'out of', train_batch_Y.shape[0])


    return correct / total,  correct, total


#Nonesense classes are different
def testAccuracy(step):

    testFile = os.path.join('D:\\','Python','Person Re ID Dataset','bbox_test', 'testV3.h5')
    hdf5_fileTest = tables.open_file(testFile, mode='r')

    frames_array = hdf5_fileTest.root.__getattr__('shuffled_frames')
    labels_array = hdf5_fileTest.root.__getattr__('shuffled_labels')

    correct = 0
    total = frames_array.shape[0]

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        saver.restore(session, 'models/model-' + str(step))

        start = 0
        batchNo = 0

        while start < total_train_size - batch_size:
            train_batch_X = frames_array[start:start + batch_size, :, :, :]
            train_batch_Y = labels_array[start:start + batch_size, :625]
            l, prediction = session.run([loss, tf_prediction],
                                        feed_dict={tf_train_X: train_batch_X, tf_train_Y: train_batch_Y})
            correctinBatch = np.sum(np.argmax(train_batch_Y, 1) == np.argmax(prediction, 1))
            print('\tBatch', batchNo, 'Correct Predictions:', correctinBatch, 'out of', train_batch_Y.shape[0])
            correct = correct + correctinBatch
            start = start + batch_size
            batchNo = batchNo + 1

        start = start - batch_size

        if start < total_train_size:
            train_batch_X = frames_array[start:, :, :, :]
            train_batch_Y = labels_array[start:, :]
            l, prediction = session.run([loss, tf_prediction],
                                        feed_dict={tf_train_X: train_batch_X, tf_train_Y: train_batch_Y})
            correctinBatch = np.sum(np.argmax(train_batch_Y, 1) == np.argmax(prediction, 1))
            print('\tBatch', batchNo, 'Correct Predictions:', correctinBatch, 'out of', train_batch_Y.shape[0])

    hdf5_fileTest.close()

    return correct / total, correct, total


def saveFeatureVectors(step, path, savePath):

    hdf5_file = tables.open_file(savePath, mode='w')
    persons = os.listdir(path)
    numPersons = len(persons)

    for per in range(numPersons):

        person_group = hdf5_file.create_group(hdf5_file.root, 'p' + persons[per], persons[per])
        dataDir = os.path.join(path, persons[per])
        loadDirRGB = os.path.join(dataDir, 'RGB')
        cameras = os.listdir(loadDirRGB)

        for cam in range(len(cameras)):

            camera_group = hdf5_file.create_group(person_group, 'c' + cameras[cam].split()[1], cameras[cam])
            loadDirRGBCam = os.path.join(loadDirRGB, cameras[cam])
            print('Processing Person ' + persons[per] + ' ' + cameras[cam])
            files = os.listdir(loadDirRGBCam)
            features_array = hdf5_file.create_earray(camera_group, 'features', img_dtype, shape=(0, 625))
            numFiles = len(files)
            frames = np.ndarray([numFiles,rows,cols,channels], np.float32)


            for i, file in enumerate(files):

                if not (len(file) > 4 and '.jpg' in file.lower() and cameras[cam].split()[1] in file):
                    continue

                seqImg = cv2.imread(os.path.join(loadDirRGBCam, file))
                seqImg = cv2.resize(seqImg, (rows, cols)) / 255
                frames[i,:,:,:] = seqImg

            maxFramestoPass = 4096
            start = 0
            while start < numFiles:

                framesToPass = frames[start:start+maxFramestoPass,:,:,:]
                start = start + maxFramestoPass

                with tf.Session(graph=graph) as session:
                    tf.global_variables_initializer().run()
                    saver.restore(session, 'models/model-' + str(step))

                    features = session.run([logits], feed_dict={tf_train_X: framesToPass})[0]
                    [features_array.append(f[None]) for f in features]

                hdf5_file.flush()

    hdf5_file.close()


def getFrequencyDist(path):

    persons = os.listdir(path)
    numPersons = len(persons)
    frequency = []

    for per in range(numPersons):

        dataDir = os.path.join(path, persons[per])
        loadDirRGB = os.path.join(dataDir, 'RGB')
        cameras = os.listdir(loadDirRGB)

        for cam in range(len(cameras)):

            loadDirRGBCam = os.path.join(loadDirRGB, cameras[cam])
            files = os.listdir(loadDirRGBCam)
            numFiles = len(files)
            frequency.append(numFiles)

    return frequency


#min max  sum   fold   avg
#6 15318 509914 1955 260.825
saveFileFeatures = os.path.join('D:\\','Python','Person Re ID Dataset','bbox_train', 'trainV3CNNFeaturesTemp.h5')
saveFeatureVectors(numIter-1, rootDir, saveFileFeatures)

#min max  sum   fold   avg
#2 13624 509966 1988 256.522
rootDirTest = os.path.join('D:\\','Python','Person Re ID Dataset','bbox_test','bbox_test')
saveFileFeaturesTest = os.path.join('D:\\','Python','Person Re ID Dataset','bbox_test', 'testV3CNNFeaturesTemp.h5')
saveFeatureVectors(numIter-1,rootDirTest, saveFileFeaturesTest)



# hdf5_file.close()