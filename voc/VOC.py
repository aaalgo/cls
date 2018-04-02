#!/usr/bin/env python3

import numpy as np

CLASSES = ['aeroplane',
           'bicycle',
           'bird',
           'boat',
           'bottle',
           'bus',
           'car',
           'cat',
           'chair',
           'cow',
           'diningtable',
           'dog',
           'horse',
           'motorbike',
           'person',
           'pottedplant',
           'sheep',
           'sofa',
           'train',
           'tvmonitor']

assert len(CLASSES) == 20

def load_list (Class, Set, difficult = True):
    # Class is the class name
    # Set can be 'train', 'val', or 'trainval'
    # set difficult to False to remove difficult examples

    # return X, Y
    # where X: list of image paths
    #       Y: labels of 0, 1

    path = 'data/VOC2012/ImageSets/Main/%s_%s.txt' % (Class, Set)
    X = []
    Y = []
    with open(path, 'r') as f:
        for line in f:
            fs = line.strip().split(' ')
            name = fs[0]
            label = fs[-1]
            label = int(label)
            if label == 0:
                if not difficult:
                    continue
                label = 1
            elif label == -1:
                label = 0
            elif label == 1:
                pass
            else:
                assert False    # unknown label
                pass
            image_path = 'data/VOC2012/JPEGImages/%s.jpg' % name
            X.append(image_path)
            Y.append(label)
            pass
        pass
    return X, np.array(Y, dtype=np.int32)

