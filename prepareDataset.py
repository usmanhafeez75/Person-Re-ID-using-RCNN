#stores data as six dimentional map
#dataset[person][camera][file][row][col][channel]

from six.moves import cPickle as pickle
import numpy as np
import cv2
import os


rows = 64
cols = 64
rootDir = os.path.join('D:\\','Python','Multiview Face Dataset','bbox_train','bbox_train')
pickleFolder = os.path.join('D:\\','Python','Multiview Face Dataset','bbox_train', 'pickles')

persons = os.listdir(rootDir)


for per in persons:

    personData = dict()
    dataDir = os.path.join(rootDir, per)
    loadDirRGB = os.path.join(dataDir, 'RGB')
    loadDirOF = os.path.join(dataDir, 'Optical Flow')

    cameras = os.listdir(loadDirRGB)

    for cam in cameras:

        loadDirRGBCam = os.path.join(loadDirRGB, cam)
        loadDirOFCam = os.path.join(loadDirOF,cam)
        print('Processing Person ' + per + ' ' + cam)

        files = os.listdir(loadDirRGBCam)
        personData[cam] = np.ndarray((len(files),rows,cols,5), float)

        for i,file in enumerate(files):

            if not (len(file) > 4 and '.jpg' in file.lower() and cam.split()[1] in file):

                continue

            seqImg = cv2.imread(os.path.join(loadDirRGBCam, file))
            seqImg = cv2.resize(seqImg, (rows,cols))
            seqImg = cv2.cvtColor(seqImg, cv2.COLOR_RGB2YUV).astype(float)

            optFl = cv2.imread(os.path.join(loadDirOFCam, file))
            optFl = cv2.resize(optFl, (rows,cols)).astype(float)

            for channel in range(3):

                seqImg[:,:,channel] = seqImg[:,:,channel] - seqImg[:,:,channel].mean()
                std = seqImg[:,:,channel].std()
                if(std != 0):
                    seqImg[:, :, channel] = seqImg[:,:,channel] / std
                optFl[:,:,channel] = optFl[:,:,channel] - optFl[:,:,channel].mean()
                std = optFl[:,:,channel].std()
                if(std != 0):
                    optFl[:, :, channel] = optFl[:,:,channel] / std

            personData[cam][i,:,:,:3] = seqImg
            personData[cam][i,:,:,3:] = optFl[:,:,:2]

    pickleFile = os.path.join(pickleFolder, per + '.pickle')
    try:
        f = open(pickleFile, 'wb')
        pickle.dump(personData, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to write data to' + pickleFile, 'wb' + ':' + e)




dataset = dict()

for per in persons:

    pickleFile = os.path.join(pickleFolder, per + '.pickle')
    with open(pickleFile, 'rb') as f:
        personData = pickle.load(f)
    dataset[per] = personData


pickleFile = os.path.join(pickleFolder, 'train.pickle')
try:
    f = open(pickleFile, 'wb')
    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    print('Pickle saved successfully')
except Exception as e:
    print('Unable to write data to' + pickleFile + ':' + e)