import numpy as np
import cv2
import os
import tables


rows = 64
cols = 64
img_dtype = tables.Float32Atom()
rootDir = os.path.join('D:\\','Python','Person Re ID Dataset','bbox_train','bbox_train')
saveFile = os.path.join('D:\\','Python','Person Re ID Dataset','bbox_train', 'trainV2.h5')

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
        frames_array = hdf5_file.create_earray(camera_group, 'frames', img_dtype, shape=(0,rows,cols,3))

        for i,file in enumerate(files):

            if not (len(file) > 4 and '.jpg' in file.lower() and cam.split()[1] in file):

                continue

            seqImg = cv2.imread(os.path.join(loadDirRGBCam, file))
            seqImg = cv2.resize(seqImg, (rows,cols))/255

            frames_array.append(seqImg[None])

    hdf5_file.flush()

hdf5_file.close()
