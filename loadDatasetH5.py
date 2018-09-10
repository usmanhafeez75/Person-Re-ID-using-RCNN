import numpy as np
import matplotlib.pyplot as plt
import os
import tables


rows = 64
cols = 64
img_dtype = tables.Float32Atom()
rootDir = os.path.join('D:\\','Python','Multiview Face Dataset','bbox_train','bbox_train')
saveFile = os.path.join('D:\\','Python','Person Re ID Dataset','bbox_train', 'train.h5')

hdf5_file = tables.open_file(saveFile, mode='r')


print(list(hdf5_file.root._v_children))
print(list(hdf5_file.root._f_get_child('p0003')._v_children))
frames = hdf5_file.root.__getattr__('p0005').__getattr__('c1').__getattr__('frames')
frames = np.array(frames)
print(type(frames),frames.shape)

# for per in hdf5_file.root:
#     for cam in per:
#         for file in cam.frames:
#             continue



hdf5_file.close()
