import cv2
import os
import tables
import numpy as np

channels = 3
rows = 64
cols = 64
img_dtype = tables.Float32Atom()
int_dtype = tables.Int16Atom()
rootDir = os.path.join('D:\\','Python','Person Re ID Dataset','bbox_test','bbox_test')
saveFile = os.path.join('D:\\','Python','Person Re ID Dataset','bbox_test', 'testV3.h5')

hdf5_file = tables.open_file(saveFile, mode='w')
persons = os.listdir(rootDir)
numPersons = len(persons)

frames_array = hdf5_file.create_earray(hdf5_file.root, 'frames', img_dtype, shape=(0,rows,cols,channels))
labels_array = hdf5_file.create_earray(hdf5_file.root, 'labels', img_dtype, shape=(0,numPersons))

for per in range(numPersons):

    dataDir = os.path.join(rootDir, persons[per])
    loadDirRGB = os.path.join(dataDir, 'RGB')

    cameras = os.listdir(loadDirRGB)

    for cam in range(len(cameras)):

        loadDirRGBCam = os.path.join(loadDirRGB, cameras[cam])
        print('Processing Person ' + persons[per] + ' ' + cameras[cam])

        files = os.listdir(loadDirRGBCam)

        for i,file in enumerate(files):

            if not (len(file) > 4 and '.jpg' in file.lower() and cameras[cam].split()[1] in file):

                continue

            seqImg = cv2.imread(os.path.join(loadDirRGBCam, file))
            seqImg = cv2.resize(seqImg, (rows,cols))/255
            frames_array.append(seqImg[None])

            label = np.zeros([numPersons])
            label[per] = 1
            labels_array.append(label[None])

    hdf5_file.flush()

permutation = np.random.permutation(frames_array.shape[0])
shuffled_frames = hdf5_file.create_earray(hdf5_file.root, 'shuffled_frames', img_dtype, shape=(0,rows,cols,channels))
shuffled_labels = hdf5_file.create_earray(hdf5_file.root, 'shuffled_labels', img_dtype, shape=(0,numPersons))

for j in permutation:

    print('Shuffling', j)
    shuffled_frames.append(frames_array[j,:,:,:][None])
    shuffled_labels.append(labels_array[j,:][None])

frames_array.remove()
labels_array.remove()

hdf5_file.close()
