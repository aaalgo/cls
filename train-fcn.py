#!/usr/bin/env python3
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'models/research/slim'))
import time
import datetime
import logging
from tqdm import tqdm
import numpy as np
import cv2
from sklearn.metrics import accuracy_score, roc_auc_score
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim
import fcn_nets as nets
import picpac

augments = None
#from . config import *
#if os.path.exists('config.py'):
def print_red (txt):
    print('\033[91m' + txt + '\033[0m')

def print_green (txt):
    print('\033[92m' + txt + '\033[0m')

print(augments)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('db', None, 'training db')
flags.DEFINE_string('val_db', None, 'validation db')
flags.DEFINE_integer('classes', 2, 'number of classes')
flags.DEFINE_string('mixin', None, 'mix-in training db')

flags.DEFINE_integer('size', None, '') 
flags.DEFINE_integer('batch', 1, 'Batch size.  ')
flags.DEFINE_integer('shift', 0, '')
flags.DEFINE_integer('stride', 16, '')

flags.DEFINE_string('net', 'myunet', 'architecture')
flags.DEFINE_string('model', None, 'model directory')
flags.DEFINE_string('resume', None, 'resume training from this model')
flags.DEFINE_integer('max_to_keep', 100, '')

# optimizer settings
flags.DEFINE_float('lr', 0.01, 'Initial learning rate.')
flags.DEFINE_float('decay_rate', 0.95, '')
flags.DEFINE_float('decay_steps', 500, '')
flags.DEFINE_float('weight_decay', 0.00004, '')
#
flags.DEFINE_integer('epoch_steps', None, '')
flags.DEFINE_integer('max_epochs', 20000, '')
flags.DEFINE_integer('ckpt_epochs', 10, '')
flags.DEFINE_integer('val_epochs', 10, '')
flags.DEFINE_boolean('adam', False, '')

COLORSPACE = 'BGR'
#PIXEL_MEANS = np.array([[[[127.0, 127.0, 127.0]]]])
#VGG_PIXEL_MEANS = np.array([[[[103.94, 116.78, 123.68]]]])
PIXEL_MEANS = tf.constant([[[[127.0, 127.0, 127.0]]]])


def fcn_loss (logits, labels):
    logits = tf.reshape(logits, (-1, FLAGS.classes))
    labels = tf.reshape(labels, (-1,))
    # cross-entropy
    xe = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    xe = tf.reduce_mean(xe, name='xe')
    # accuracy
    acc = tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32)
    acc = tf.reduce_mean(acc, name='acc')
    # regularization
    reg = tf.reduce_sum(tf.losses.get_regularization_losses())
    reg = tf.identity(reg, name='re')
    # loss
    loss = tf.identity(xe + reg, name='lo')
    return loss, [acc, xe, reg, loss]

def create_picpac_stream (db_path, is_training):
    assert os.path.exists(db_path)
    augments = []
    if is_training:
        augments = [
                  #{"type": "augment.flip", "horizontal": True, "vertical": False},
                  {"type": "augment.rotate", "min":-10, "max":10},
                  {"type": "augment.scale", "min":0.9, "max":1.1},
                  {"type": "augment.add", "range":20},
                ]
    else:
        augments = []

    config = {"db": db_path,
              "loop": is_training,
              "shuffle": is_training,
              "reshuffle": is_training,
              "annotate": True,
              "channels": 3,
              "stratify": is_training,
              "dtype": "float32",
              "batch": FLAGS.batch,
              "colorspace": COLORSPACE,
              "transforms": augments + [
                  {"type": "clip", "round": FLAGS.stride},
                  {"type": "rasterize"},
                  ]
             }
    if is_training and not FLAGS.mixin is None:
        print("mixin support is incomplete in new picpac.")
    #    assert os.path.exists(FLAGS.mixin)
    #    picpac_config['mixin'] = FLAGS.mixin
    #    picpac_config['mixin_group_delta'] = 1
    #    pass
    return picpac.ImageStream(config)

def main (_):
    global PIXEL_MEANS

    logging.basicConfig(filename='train-%s-%s.log' % (FLAGS.net, datetime.datetime.now().strftime('%Y%m%d-%H%M%S')),level=logging.DEBUG, format='%(asctime)s %(message)s')

    if FLAGS.model:
        try:
            os.makedirs(FLAGS.model)
        except:
            pass

    X = tf.placeholder(tf.float32, shape=(None, None, None, 3), name="images")

    # ground truth labels
    Y = tf.placeholder(tf.int32, shape=(None, None, None, 1), name="labels")
    is_training = tf.placeholder(tf.bool, name="is_training")

    #with \
    #     slim.arg_scope([slim.batch_norm], decay=0.9, epsilon=5e-4): 

    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.max_pool2d],
                            padding='SAME'), \
                                    slim.arg_scope([slim.conv2d, slim.conv2d_transpose], weights_regularizer=slim.l2_regularizer(2.5e-4), normalizer_fn=slim.batch_norm, normalizer_params={'decay': 0.9, 'epsilon': 5e-4, 'scale': False, 'is_training':is_training}), \
         slim.arg_scope([slim.batch_norm], is_training=is_training):
        logits, FLAGS.stride = getattr(nets, FLAGS.net)(X-PIXEL_MEANS)

    # probability of class 1 -- not very useful if FLAGS.classes > 2
    probs = tf.squeeze(tf.slice(tf.nn.softmax(logits), [0,0,0,1], [-1,-1,-1,1]), 3)

    loss, metrics = fcn_loss(logits, Y)
    metric_names = [x.name[:-2] for x in metrics]

    def format_metrics (avg):
        return ' '.join(['%s=%.3f' % (a, b) for a, b in zip(metric_names, list(avg))])

    global_step = tf.train.create_global_step()
    LR = tf.train.exponential_decay(FLAGS.lr, global_step, FLAGS.decay_steps, FLAGS.decay_rate, staircase=True)
    if FLAGS.adam:
        print("Using Adam optimizer, reducing LR by 100x")
        optimizer = tf.train.AdamOptimizer(LR/100)
    else:
        optimizer = tf.train.MomentumOptimizer(learning_rate=LR, momentum=0.9)

    train_op = slim.learning.create_train_op(loss, optimizer, global_step=global_step)
    saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)

    stream = create_picpac_stream(FLAGS.db, True)
    # load validation db
    val_stream = None
    if FLAGS.val_db:
        val_stream = create_picpac_stream(FLAGS.val_db, False)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    epoch_steps = FLAGS.epoch_steps
    if epoch_steps is None:
        epoch_steps = (stream.size() + FLAGS.batch-1) // FLAGS.batch
    best = 0
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        if FLAGS.resume:
            saver.restore(sess, FLAGS.resume)

        global_start_time = time.time()
        epoch = 0
        step = 0
        while epoch < FLAGS.max_epochs:
            start_time = time.time()
            cnt, metrics_sum = 0, np.array([0] * len(metrics), dtype=np.float32)
            progress = tqdm(range(epoch_steps), leave=False)
            for _ in progress:
                _, images, labels = stream.next()
                feed_dict = {X: images, Y: labels, is_training: True}
                mm, _ = sess.run([metrics, train_op], feed_dict=feed_dict)
                metrics_sum += np.array(mm) * images.shape[0]
                cnt += images.shape[0]
                metrics_txt = format_metrics(metrics_sum/cnt)
                progress.set_description(metrics_txt)
                step += 1
                pass
            stop = time.time()
            msg = 'train epoch=%d step=%d ' % (epoch, step)
            msg += metrics_txt
            msg += ' elapsed=%.3f time=%.3f ' % (stop - global_start_time, stop - start_time)
            print_green(msg)
            logging.info(msg)

            epoch += 1

            if (epoch % FLAGS.val_epochs == 0) and val_stream:
                lr = sess.run(LR)
                # evaluation
                Ys, Ps = [], []
                cnt, metrics_sum = 0, np.array([0] * len(metrics), dtype=np.float32)
                val_stream.reset()
                progress = tqdm(val_stream, leave=False)
                for _, images, labels in progress:
                    feed_dict = {X: images, Y: labels, is_training: False}
                    p, mm = sess.run([probs, metrics], feed_dict=feed_dict)
                    metrics_sum += np.array(mm) * images.shape[0]
                    cnt += images.shape[0]
                    Ys.extend(list(meta.labels))
                    Ps.extend(list(p))
                    metrics_txt = format_metrics(metrics_sum/cnt)
                    progress.set_description(metrics_txt)
                    pass
                assert cnt == val_stream.size()
                avg = metrics_sum / cnt
                if avg[0] > best:
                    best = avg[0]
                msg = 'valid epoch=%d step=%d ' % (epoch-1, step)
                msg += metrics_txt
                if FLAGS.classes == 2:
                    # display scikit-learn metrics
                    Ys = np.array(Ys, dtype=np.int32)
                    Ps = np.array(Ps, dtype=np.float32)
                    msg += ' sk_acc=%.3f auc=%.3f' % (accuracy_score(Ys, Ps > 0.5), roc_auc_score(Ys, Ps))
                    pass
                msg += ' lr=%.4f best=%.3f' % (lr, best)
                print_red(msg)
                logging.info(msg)
                #log.write('%d\t%s\t%.4f\n' % (epoch, '\t'.join(['%.4f' % x for x in avg]), best))
            # model saving
            if (epoch % FLAGS.ckpt_epochs == 0) and FLAGS.model:
                ckpt_path = '%s/%d' % (FLAGS.model, epoch)
                saver.save(sess, ckpt_path)
                print('saved to %s.' % ckpt_path)
            pass
        pass
    pass

if __name__ == '__main__':
    try:
        tf.app.run()
    except KeyboardInterrupt:
        pass

