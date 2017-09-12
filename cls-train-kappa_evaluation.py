#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pkgutil
import numpy as np
import os
import datetime
import picpac
import tensorflow as tf
import tensorflow.contrib.slim as slim
#from tensorflow.python.client import timeline

# --net=module.model
# where module is the python file basename of one of these
#   https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim/python/slim/nets
# and model is the function name within the file defining the net.
# e.g. the following have worked for very small datasets
#   alexnet.alexnet_v2
#   inception_v3.inception_v3
#   vgg.vgg_16

# Failed to converge:
#   vgg.vgg_a
#   inception_v1.inception_v1
#
# Not working yet:  resnet for requirement of special API (blocks)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('opt','adam', '')
flags.DEFINE_string('db', 'db', 'database')
flags.DEFINE_string('test_db', None, 'evaluation dataset')
flags.DEFINE_integer('classes', '2', 'number of classes')
flags.DEFINE_integer('resize', None, '')
flags.DEFINE_integer('channels', 3, '')
flags.DEFINE_integer('batch', 1, 'Batch size.  ')
flags.DEFINE_string('net', 'resnet_v1.resnet_v1_50', 'cnn architecture, e.g. vgg.vgg_a')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('test_steps', 1000, 'Number of steps to run evaluation.')
flags.DEFINE_integer('save_steps', 1000, 'Number of steps to run evaluation.')
flags.DEFINE_integer('max_steps', 800000, 'Number of steps to run trainer.')
flags.DEFINE_string('model', 'model', 'Directory to put the training data.')
flags.DEFINE_integer('split', 1, 'split into this number of parts for cross-validation')
flags.DEFINE_integer('split_fold', 0, 'part index for cross-validation')

# load network architecture by name
def inference (inputs, num_classes):
    full = 'tensorflow.contrib.slim.python.slim.nets.' + FLAGS.net
    # e.g. full == 'tensorflow.contrib.slim.python.slim.nets.vgg.vgg_a'
    fs = full.split('.')
    loader = pkgutil.find_loader('.'.join(fs[:-1]))
    module = loader.load_module('')
    net = getattr(module, fs[-1])
    logits, _ = net(inputs, num_classes)
    logits = tf.squeeze(logits, [1,2]) # resnet output is (N,1,1,C, remove the 
    return tf.identity(logits, name='logits')

def fcn_loss (logits, labels):
    with tf.name_scope('loss'):
        labels = tf.to_int32(labels)    # float from picpac
        xe = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy')
        hit = tf.cast(tf.nn.in_top_k(logits, labels, 1, name="accuracy"), tf.float32)
        return [tf.reduce_mean(xe, name='xentropy_mean'), tf.reduce_mean(hit, name='accuracy_total')]
    pass

def training (loss, rate):
    #tf.scalar_summary(loss.op.name, loss)
	if FLAGS.opt == 'adam':
		rate /= 100
		optimizer = tf.train.AdamOptimizer(rate)
		print('adam!')
	else:
		optimizer = tf.train.GradientDescentOptimizer(rate)
		print('gradient!')
		pass
   
	global_step = tf.Variable(0, name='global_step', trainable=False)
	return optimizer.minimize(loss, global_step=global_step)

def run_training ():
    try:
        os.makedirs(FLAGS.model)
    except:
        pass
    config = dict(seed=1996,
                shuffle=True,
                reshuffle=True,
                #resize_width=FLAGS.resize,
                #resize_height=FLAGS.resize,
                batch=FLAGS.batch,
                split=FLAGS.split,
                split_fold=FLAGS.split_fold,
                channels=FLAGS.channels,
                stratify=True,
                #mixin="db0",
                #mixin_group_delta=0,
                pert_color1=10,
                pert_color2=10,
                pert_color3=10,
                #pert_angle=10,
                pert_min_scale=0.8,
                pert_max_scale=1.2,
                #pad=False,
                #pert_hflip=True,
                channel_first=False # this is tensorflow specific
                                    # Caffe's dimension order is different.
                )
    # training stream
    tr_stream = picpac.ImageStream(FLAGS.db, split_negate=False, perturb=False, loop=True, **config)
    te_stream = None
    if FLAGS.test_steps > 0:
        # testing stream, "negate" inverts the image selection specified by split & split_fold
        # so different images are used for training and testing
        if FLAGS.test_db:
            if FLAGS.split > 1:
                print("Cannot use cross-validation & evaluation db at the same time")
                print("If --test-db is specified, do not set --split")
                raise Exception("bad parameters")
            te_stream = picpac.ImageStream(FLAGS.test_db, perturb=False, loop=False, **config)
        elif FLAGS.split > 1:
            te_stream = picpac.ImageStream(FLAGS.db, split_negate=True, perturb=False, loop=False, **config)
            pass
        pass

    with tf.Graph().as_default():
        X = tf.placeholder(tf.float32, shape=(FLAGS.batch, None, None, FLAGS.channels), name="images")
        Y_ = tf.placeholder(tf.float32, shape=(FLAGS.batch,), name="labels")
        logits = inference(X, FLAGS.classes)


        loss, accuracy = fcn_loss(logits, Y_)
        train_op = training(loss, FLAGS.learning_rate)
        #summary_op = tf.merge_all_summaries()
        #summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, tf.get_default_graph())

        init = tf.global_variables_initializer()

        #graph_txt = tf.get_default_graph().as_graph_def().SerializeToString()
        #with open(os.path.join(FLAGS.train_dir, "graph"), "w") as f:
        #    f.write(graph_txt)
        #    pass

        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #run_metadata = tf.RunMetadata()
        loss_sum = 0
        accuracy_sum = 0
        batch_sum = 0
        with tf.Session(config=config) as sess:
            sess.run(init)
            for step in xrange(FLAGS.max_steps):
                images, labels, pad = tr_stream.next()
                #print(images.shape, labels.shape)
                feed_dict = {X: images,
                             Y_: labels}
                #l_v, s_v = sess.run([logits, score], feed_dict=feed_dict)
                #print(images.shape, s_v.shape, l_v.shape)
                #_, loss_value = sess.run([train_op, loss], feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
                _, loss_value, accuracy_value, ll = sess.run([train_op, loss, accuracy, logits], feed_dict=feed_dict)
                #print('XXX', labels[0], accuracy_value, ll[0])
                loss_sum += loss_value * FLAGS.batch
                accuracy_sum += accuracy_value * FLAGS.batch
                batch_sum += FLAGS.batch
                if (step + 1) % 1000 == 0:
                    #tl = timeline.Timeline(run_metadata.step_stats)
                    #ctf = tl.generate_chrome_trace_format()
                    #with open('timeline.json', 'w') as f:
                    #    f.write(ctf)

                    print(datetime.datetime.now())
                    print('step %d: loss = %.4f, accuracy = %.4f' % (step+1, loss_sum/batch_sum, accuracy_sum/batch_sum))
                    loss_sum = 0
                    accuracy_sum = 0
                    batch_sum = 0
                    #summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    #summary_writer.add_summary(summary_str, step)
                    #summary_writer.flush()
                if te_stream and (step + 1) % FLAGS.test_steps == 0:
                    # evaluation
                    te_stream.reset()
                    cmatrix = np.zeros((FLAGS.classes,FLAGS.classes))
                    total = 0
                    batch_sum2 = 0
                    loss_sum2 = 0
                    accuracy_sum2 = 0
                    for images, labels, pad in te_stream:
                        bs = FLAGS.batch - pad
                        total += 1
                        if pad > 0:
                            numpy.resize(images, (bs,)+images.shape[1:])
                            numpy.resize(labels, (bs,))

                        feed_dict = {X: images,
                                     Y_: labels}

                        loss_value, accuracy_value,ll = sess.run([loss, accuracy,logits], feed_dict=feed_dict)

                        batch_sum2 += bs
                        cmatrix[int(labels), np.where(ll == np.max(ll))[1]] += np.divide(float(1),np.size(np.where(ll == np.max(ll))[1]))
                        loss_sum2 += loss_value * bs
                        #print(int(labels))
#print(cmatrix)
                        #print(np.size(np.where(ll == np.max(ll))[1]))
                        #print(ll)
                        accuracy_sum2 += accuracy_value * bs
						
                        pass
                    rowsum = np.sum(cmatrix, axis = 1)
                    colsum = np.sum(cmatrix, axis = 0)
#print(total)
#print(np.tile(rowsum,(FLAGS.classes,1)).transpose())
#print(np.tile(colsum,(FLAGS.classes,1)))
                    print('row---label; colum ---predict')
                    print(total)
                    print('evaluation: loss = %.4f, accuracy = %.4f' % (loss_sum2/batch_sum2, accuracy_sum2/batch_sum2))
                    print('accuracy from confusion matrix = %.4f' % np.divide(np.trace(cmatrix),float(total)))
                    print('absolute confusion matrix:') 
                    print(cmatrix)
                    #print('confusion matrix divided by total:')
                    #print(np.divide(cmatrix,float(total)))
                    print('confusion matrix divided by col sum:')
                    print(np.divide(cmatrix,np.tile(colsum,(5,1))))
                    print('confusion matrix divided by row sum:')
                    print(np.divide(cmatrix,np.tile(rowsum,(5,1)).transpose()))
                    print('Weighted kappa quadratic scores:')
                    print(quadratic_kappa(cmatrix))
                if (step + 1) % FLAGS.save_steps == 0 or (step + 1) == FLAGS.max_steps:
                    ckpt_path = '%s/%d' % (FLAGS.model, (step + 1))
                    saver.save(sess, ckpt_path)
                pass
            pass
        pass
    pass

def quadratic_kappa(y):
    # num_scored_items = y.shape[0]
    num_scored_items = y.sum()
    num_ratings = y.shape[0]
    # num_ratings = y.shape[1]
    ratings_mat = np.tile(np.arange(0, num_ratings)[:, None],
            reps=(1, num_ratings))
    ratings_squared = (ratings_mat - ratings_mat.T)**2
    weights = ratings_squared / (float(num_ratings)-1)**2

    # y_norm = y/(np.divide(y.sum(axis=1)[:, None]), float(1))

    hist_rater_a = y.sum(axis=1)
    hist_rater_b = y.sum(axis=0)

    # conf_mat = np.dot(y_norm.T, t)
    nom = np.sum(weights * y)
    expected_prob = np.dot(hist_rater_a[:, None], 
            hist_rater_b[None, :])
    # denom = np.sum(weights * expected_probs/num_scored_items)
    denom = np.sum(weights * expected_prob/num_scored_items)
    return 1 - nom/denom

def main (_):
    run_training()

if __name__ == '__main__':
    tf.app.run()

