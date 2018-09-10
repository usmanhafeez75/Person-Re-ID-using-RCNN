import numpy as np
import cv2
import os
import tables


rows = 64
cols = 64
img_dtype = tables.Float32Atom()
rootDir = os.path.join('D:\\','Python','Person Re ID Dataset','bbox_train','bbox_train')
saveFile = os.path.join('D:\\','Python','Person Re ID Dataset','bbox_train', 'train.h5')

hdf5_file = tables.open_file(saveFile, mode='w')
persons = os.listdir(rootDir)


for per in persons:

    person_group = hdf5_file.create_group(hdf5_file.root, 'p' + per, per)
    dataDir = os.path.join(rootDir, per)
    loadDirRGB = os.path.join(dataDir, 'RGB')
    loadDirOF = os.path.join(dataDir, 'Optical Flow')

    cameras = os.listdir(loadDirRGB)

    for cam in cameras:

        camera_group = hdf5_file.create_group(person_group, 'c' + cam.split()[1], cam)
        loadDirRGBCam = os.path.join(loadDirRGB, cam)
        loadDirOFCam = os.path.join(loadDirOF,cam)
        print('Processing Person ' + per + ' ' + cam)

        files = os.listdir(loadDirRGBCam)
        frames_array = hdf5_file.create_earray(camera_group, 'frames', img_dtype, shape=(0,rows,cols,5))
        means_array = hdf5_file.create_earray(camera_group, 'means', tables.Float32Atom(), shape=(0,2,3))
        stds_array = hdf5_file.create_earray(camera_group, 'stds', tables.Float32Atom(), shape=(0,2,3))

        for i,file in enumerate(files):

            if not (len(file) > 4 and '.jpg' in file.lower() and cam.split()[1] in file):

                continue

            seqImg = cv2.imread(os.path.join(loadDirRGBCam, file))
            seqImg = cv2.resize(seqImg, (rows,cols))
            seqImg = cv2.cvtColor(seqImg, cv2.COLOR_RGB2YUV).astype(float)

            optFl = cv2.imread(os.path.join(loadDirOFCam, file))
            optFl = cv2.resize(optFl, (rows,cols)).astype(float)

            means = np.ndarray((2,3), float)
            stds = np.ndarray((2,3), float)

            for channel in range(3):

                means[0,channel] =  seqImg[:,:,channel].mean()
                seqImg[:,:,channel] = seqImg[:,:,channel] - means[0, channel]
                stds[0, channel] = seqImg[:,:,channel].std()
                if(stds[0, channel] != 0):
                    seqImg[:, :, channel] = seqImg[:,:,channel] / stds[0, channel]
                means[1, channel] = optFl[:,:,channel].mean()
                optFl[:,:,channel] = optFl[:,:,channel] - means[1, channel]
                stds[1, channel] = optFl[:,:,channel].std()
                if(stds[1, channel] != 0):
                    optFl[:, :, channel] = optFl[:,:,channel] / stds[0, channel]

            frame = np.ndarray((rows,cols,5), float)
            frame[:,:,:3] = seqImg
            frame[:,:,3:] = optFl[:,:,:2]
            frames_array.append(frame[None])
            means_array.append(means[None])
            stds_array.append(stds[None])

    hdf5_file.flush()

hdf5_file.close()
