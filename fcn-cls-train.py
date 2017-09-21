#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import logging
from tqdm import tqdm
from skimage import measure
from random import randint
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils
import picpac
import fcn_cls_nets

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('pos', 'db.pos', 'training dataset')
flags.DEFINE_string('neg', 'db.neg', 'training dataset')
flags.DEFINE_string('test_db', None, 'evaluation dataset')
#flags.DEFINE_string('mixin', None, 'mixin negative dataset')

flags.DEFINE_string('model', 'model', 'Directory to put the training data.')
flags.DEFINE_string('net', 'resnet_tiny', '')

flags.DEFINE_string('opt', 'adam', '')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_bool('decay', True, '')
flags.DEFINE_float('decay_rate', 0.9, '')
flags.DEFINE_float('decay_steps', 10000, '')
flags.DEFINE_float('momentum', 0.99, 'when opt==mom')
flags.DEFINE_string('resume', None, '')
flags.DEFINE_integer('max_steps', 200000, '')
flags.DEFINE_integer('epoch_steps', 100, '')
flags.DEFINE_integer('test_epochs', 10, '')
flags.DEFINE_integer('ckpt_epochs', 200, '')
#flags.DEFINE_string('log', None, 'tensorboard')
flags.DEFINE_integer('channels', 3, '')
flags.DEFINE_string('padding', 'SAME', '')
flags.DEFINE_integer('verbose', logging.INFO, '')
flags.DEFINE_float('pos_weight', None, '')
flags.DEFINE_integer('max_size', None, '')
flags.DEFINE_integer('max_to_keep', 10, '')
flags.DEFINE_integer('split', 1, 'split into this number of parts for cross-validation')
flags.DEFINE_integer('split_fold', 1, 'part index for cross-validation')
flags.DEFINE_float('W', 1.0, '')
MAX_SAMPLES = 100


def fcn_loss (logits, labels):
    # to HWC
    logits = tf.reshape(logits, (-1, 2))
    labels = tf.reshape(labels, (-1,))
    xe = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.to_int32(labels))
    if FLAGS.pos_weight:
        POS_W = tf.pow(tf.constant(FLAGS.pos_weight, dtype=tf.float32),
                       labels)
        xe = tf.multiply(xe, POS_W)
    loss = tf.reduce_mean(xe, name='fcn_xe')
    return loss

def cls_loss (logits, labels):
    # to HWC
    logits = tf.reshape(logits, (-1, 2))
    labels = tf.to_int32(tf.reshape(labels, (-1,)))
    xe = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(xe, name='cls_xe')
    acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32), name='cls_acc')
    return loss, acc

def main (_):
    logging.basicConfig(level=FLAGS.verbose)
    try:
        os.makedirs(FLAGS.model)
    except:
        pass
    assert FLAGS.pos and os.path.exists(FLAGS.pos)
    assert FLAGS.neg and os.path.exists(FLAGS.neg)

    X = tf.placeholder(tf.float32, shape=(None, None, None, FLAGS.channels), name="images")
    Y_fcn = tf.placeholder(tf.float32, shape=(None, None, None, 1))
    Y_cls = tf.placeholder(tf.float32, shape=(None,))

    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.max_pool2d],
                            padding=FLAGS.padding):
        logits_fcn, logits_cls, stride = getattr(fcn_cls_nets, FLAGS.net)(X)
    loss_fcn = fcn_loss(logits_fcn, Y_fcn)
    loss_cls, accuracy_cls = cls_loss(logits_cls, Y_cls)

    #tf.summary.scalar("loss", loss)
    loss_total = tf.identity(loss_fcn + FLAGS.W * loss_cls, name='loss')

    metrics = [loss_total, loss_fcn, loss_cls, accuracy_cls]
    metric_names = [x.name[:-2] for x in metrics]

    rate = FLAGS.learning_rate
    if FLAGS.opt == 'adam':
        rate /= 100
    global_step = tf.Variable(0, name='global_step', trainable=False)
    if FLAGS.decay:
        rate = tf.train.exponential_decay(rate, global_step, FLAGS.decay_steps, FLAGS.decay_rate, staircase=True)
    if FLAGS.opt == 'adam':
        optimizer = tf.train.AdamOptimizer(rate)
    elif FLAGS.opt == 'mom':
        optimizer = tf.train.MomentumOptimizer(rate, FLAGS.momentum)
    else:
        optimizer = tf.train.GradientDescentOptimizer(rate)
        pass


    train_op = optimizer.minimize(loss_total, global_step=global_step)

    picpac_config_cls = dict(seed=2016,
                shuffle=True,
                reshuffle=True,
                #max_size = 300,
                #resize_width=256,
                #resize_height=256,
                split=FLAGS.split,
                split_fold=FLAGS.split_fold,
                batch=1,
                pert_angle=15,
                pert_hflip=True,
                pert_vflip=False,
                pert_color1=20,
                pert_color2=20,
                pert_color3=20,
                pert_min_scale = 0.8,
                pert_max_scale = 1.2,
                channels=FLAGS.channels,
                #mixin = FLAGS.mixin,
                stratify=True,
                #pad=False,
                channel_first=False, # this is tensorflow specific
                )
    picpac_config_fcn = dict(
                annotate='json',
                round_div = stride,
                )
    picpac_config_fcn.update(picpac_config_cls)

    tr_stream0 = picpac.ImageStream(FLAGS.neg, split_negate=False, perturb=True, loop=True, **picpac_config_fcn)
    tr_stream1 = picpac.ImageStream(FLAGS.pos, split_negate=False, perturb=True, loop=True, **picpac_config_fcn)
    te_streams = []
    if FLAGS.test_epochs > 0:
        # testing stream, "negate" inverts the image selection specified by split & split_fold
        # so different images are used for training and testing
        if FLAGS.test_db:
            if FLAGS.split > 1:
                print("Cannot use cross-validation & evaluation db at the same time")
                print("If --test-db is specified, do not set --split")
                raise Exception("bad parameters")
            te_streams.append((0, picpac.ImageStream(FLAGS.test_db, perturb=False, loop=False, **picpac_config_cls)))
        elif FLAGS.split > 1:
            te_streams.append((0, picpac.ImageStream(FLAGS.neg, split_negate=True, perturb=False, loop=False, **picpac_config_cls)))
            te_streams.append((1, picpac.ImageStream(FLAGS.pos, split_negate=True, perturb=False, loop=False, **picpac_config_cls)))
            pass
        pass


    init = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3


    with tf.Session(config=config) as sess:
        sess.run(init)
        if FLAGS.resume:
            saver.restore(sess, FLAGS.resume)
        step = 0
        epoch = 0
        global_start_time = time.time()
        while step < FLAGS.max_steps:
            start_time = time.time()
            avg = np.array([0] * len(metrics), dtype=np.float32)
            for _ in tqdm(range(FLAGS.epoch_steps), leave=False):
                # train FCN
                images, labels, _ = tr_stream0.next()
                #print('xxx', images.shape)
                feed_dict = {X: images, Y_fcn: labels, Y_cls:[0]}
                mm, _ = sess.run([metrics, train_op], feed_dict=feed_dict)
                avg += np.array(mm)

                images, labels, _ = tr_stream1.next()
                #print('xxx', images.shape)
                feed_dict = {X: images, Y_fcn: labels, Y_cls:[1]}
                mm, _ = sess.run([metrics, train_op], feed_dict=feed_dict)
                avg += np.array(mm)

                step += 1
                pass
            avg /= FLAGS.epoch_steps * 2
            stop_time = time.time()
            txt = ', '.join(['%s=%.4f' % (a, b) for a, b in zip(metric_names, list(avg))])
            print('step %d: elapsed=%.4f time=%.4f, %s'
                    % (step, (stop_time - global_start_time), (stop_time - start_time), txt))
            epoch += 1
            if epoch and (epoch % FLAGS.ckpt_epochs == 0):
                ckpt_path = '%s/%d' % (FLAGS.model, step)
                start_time = time.time()
                saver.save(sess, ckpt_path)
                stop_time = time.time()
                print('epoch %d step %d, saving to %s in %.4fs.' % (epoch, step, ckpt_path, stop_time - start_time))
            if epoch and (epoch % FLAGS.test_epochs == 0) and len(te_streams) > 0:
                cmatrix = np.zeros((2, 2))
                total = 0
                acc_sum = 0
                for delta, te_stream in te_streams:
                    te_stream.reset()
                    for images, labels, _ in te_stream:
                        total += 1
                        labels += delta

                        feed_dict = {X: images, Y_cls: labels}

                        acc, ll = sess.run([accuracy_cls, logits_cls], feed_dict=feed_dict)
                        acc_sum += acc

                        cmatrix[int(labels), np.where(ll == np.max(ll))[1]] += np.divide(float(1),np.size(np.where(ll == np.max(ll))[1]))
                        pass
                rowsum = np.sum(cmatrix, axis = 1)
                colsum = np.sum(cmatrix, axis = 0)
#print(total)
#print(np.tile(rowsum,(FLAGS.classes,1)).transpose())
#print(np.tile(colsum,(FLAGS.classes,1)))
                print('row---label; colum ---predict')
                print(total)
                print('evaluation: accuracy = %.4f' % (acc_sum/total))
                print('accuracy from confusion matrix = %.4f' % np.divide(np.trace(cmatrix),float(total)))
                print('absolute confusion matrix:') 
                print(cmatrix)
                #print('confusion matrix divided by total:')
                #print(np.divide(cmatrix,float(total)))
                print('confusion matrix divided by col sum:')
                print(np.divide(cmatrix,np.tile(colsum,(2,1))))
                print('confusion matrix divided by row sum:')
                print(np.divide(cmatrix,np.tile(rowsum,(2,1)).transpose())) 
            pass
        pass
    pass

if __name__ == '__main__':
    tf.app.run()

