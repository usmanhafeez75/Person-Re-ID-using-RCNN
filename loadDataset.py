from prepareDataset import pickleFile,pickle

with open(pickleFile, 'rb') as f:
    dataset = pickle.load(f)

for per in dataset:
    for cam in dataset[per]:
        print(dataset[per][cam])