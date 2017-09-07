#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pkgutil
import numpy as np
import cv2
import os
import datetime
import tensorflow as tf
from tensorflow.python.framework import meta_graph
import argparse

# tensorflow model
class Model:
    def __init__ (self, X, path, name, node='logits:0', softmax=True):
        """applying tensorflow image model.

        path -- path to model
        name -- output tensor name
        prob -- convert output (softmax) to probability
        """
        mg = meta_graph.read_meta_graph_file(path + '.meta')
        output, = tf.import_graph_def(mg.graph_def, name=name,
                    input_map={'images:0': X},
                    return_elements=[node])
        if softmax:
            output = tf.nn.softmax(output)
        self.output = output
        self.saver = tf.train.Saver(saver_def=mg.saver_def, name=name)
        self.load = lambda sess: self.saver.restore(sess, path)
        pass
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='PROG')
    parser.add_argument('--input', help='input directory')
    parser.add_argument('--model', default=None ,help='use model to make prediction')
    parser.add_argument('--channels', type=int, default=3, help='channels')

    args = parser.parse_args()
    assert not args.model is None
    
    X = tf.placeholder(tf.float32, shape=(None, None, None, None))
    model = Model(X, args.model, 'model')

    imread_color = -1
    if args.channels == 3:
        imread_color = cv2.IMREAD_COLOR
    elif args.channels == 1:
        imread_color = cv2.IMREAD_GRAYSCALE
        pass

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.load(sess)
        # recursively find files from args.input
        for root, dirnames, filenames in os.walk(args.input):
            for filename in filenames:
                path = os.path.join(root, filename)
                image = cv2.imread(path, imread_color)
                x = np.expand_dims(image, axis=0)
                if len(x.shape) == 3:   # gray image, add another dimension in the end
                    x = np.expand_dims(x, axis=3)
                y = sess.run(model.output, feed_dict={X: x})
                print(path)
                print('\t', y[0])
                pass
        pass
    pass

