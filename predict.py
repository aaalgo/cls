#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
from tqdm import tqdm
import numpy as np
import cv2
from skimage import measure
# RESNET: import these for slim version of resnet
import tensorflow as tf
from tensorflow.python.framework import meta_graph
import scipy
import collections

class Model:
    def __init__ (self, X, is_training, path, name):
        mg = meta_graph.read_meta_graph_file(path + '.meta')
        self.logits, = tf.import_graph_def(mg.graph_def, name=name,
                    input_map={'images:0': X, 'is_training:0': is_training},
                    return_elements=['logits:0'])
        if len(self.logits.get_shape()) == 4:
            # FCN
            self.is_fcn = True
            self.prob = tf.squeeze(tf.slice(tf.nn.softmax(self.logits), [0,0,0,1], [-1,-1,-1,1]), 3)
        else:
            # classification
            self.is_fcn = False
            self.prob = tf.nn.softmax(self.logits)
        self.saver = tf.train.Saver(saver_def=mg.saver_def, name=name)
        self.loader = lambda sess: self.saver.restore(sess, path)
        pass
    pass

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model', None, '')
flags.DEFINE_string('input', None, '')
flags.DEFINE_string('name', 'logits:0', '')
flags.DEFINE_float('cth', 0.5, '')
flags.DEFINE_integer('stride', 16, '')
flags.DEFINE_float('mean_pixel', 127, '')


def save_prediction_image (path, image, prob):
    # image: original input image
    # prob: probability

    contours = measure.find_contours(prob, FLAGS.cth)

    prob *= 255
    prob = cv2.cvtColor(prob, cv2.COLOR_GRAY2BGR)

    for contour in contours:
        tmp = np.copy(contour[:,0])
        contour[:, 0] = contour[:, 1]
        contour[:, 1] = tmp
        contour = contour.reshape((1, -1, 2)).astype(np.int32)
        cv2.polylines(image, contour, True, (0, 255,0))
        cv2.polylines(prob, contour, True, (0,255,0))

    H = max(image.shape[0], prob.shape[0])
    both = np.zeros((H, image.shape[1] + prob.shape[1], 3))
    both[0:image.shape[0],:image.shape[1], :] = image
    both[0:prob.shape[0],image.shape[1]:, :] = prob
    cv2.imwrite(path, both)

def main (_):
    assert os.path.exists(FLAGS.input)
    X = tf.placeholder(tf.float32, shape=(None, None, None, 3), name="images")
    is_training = tf.placeholder(tf.bool, name="is_training")
    model = Model(X, is_training, FLAGS.model, 'xxx')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        model.loader(sess)
        image = cv2.imread(FLAGS.input, -1)
        if model.is_fcn:    # clip image to multiple of strides
            H, W = image.shape[:2]
            H = H // FLAGS.stride * FLAGS.stride
            W = W // FLAGS.stride * FLAGS.stride
            image = image[:H, :W, :]
        batch = np.expand_dims(image, axis=0).astype(dtype=np.float32)
        batch -= FLAGS.mean_pixel
        prob = sess.run(model.prob, feed_dict={X: batch, is_training: False})
        if len(prob.shape) == 1:
            print(prob[0])
        else:
            save_prediction_image(FLAGS.input + '.prob.png', image, prob[0])
        pass
    pass

if __name__ == '__main__':
    tf.app.run()

