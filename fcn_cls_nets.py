#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_utils

def resnet_tiny (inputs, num_classes=2, scope ='resnet_tiny'):
    blocks = [ 
        resnet_utils.Block('block1', resnet_v1.bottleneck,
                           [(64, 32, 1)] + [(64, 32, 2)]),
        resnet_utils.Block('block2', resnet_v1.bottleneck,
                           [(128, 64, 1)] + [(128, 64, 2)]),
        resnet_utils.Block('block3', resnet_v1.bottleneck,
                           [(256, 64, 1)] + [(128, 64, 2)]),
        resnet_utils.Block('block4', resnet_v1.bottleneck, [(128, 64, 1)])
    	]   
    net,_ = resnet_v1.resnet_v1(
        inputs, blocks,
        # all parameters below can be passed to resnet_v1.resnet_v1_??
        num_classes = None,       # don't produce final prediction
        global_pool = False,       # produce 1x1 output, equivalent to input of a FC layer
        output_stride = 16,
        include_root_block=True,
        reuse=False,              # do not re-use network
        scope=scope)
    res_out = net		  # keep this for later CLS usage
    net = slim.batch_norm(slim.conv2d_transpose(net, 64, 5, 2))
    net = slim.batch_norm(slim.conv2d_transpose(net, 32, 5, 2))
    net = slim.batch_norm(slim.conv2d_transpose(net, 16, 5, 2))
    net = slim.batch_norm(slim.conv2d_transpose(net, 8, 5, 2))
    net = slim.conv2d(net, num_classes, 5, 1, activation_fn=None) 
    logits_fcn = tf.identity(net, 'logits_fcn')
    
    net = res_out
    # add a few layers to make the image size even smaller
    net = slim.conv2d(net, 128, 3, 1)
    net = slim.max_pool2d(net, 2, 2)
    net = slim.conv2d(net, 128, 3, 1)
    net = slim.max_pool2d(net, 2, 2)
    net = tf.reduce_mean(net, [1, 2], keep_dims=True)
    # add an extra layer
    net = slim.conv2d(net, 64, [1, 1])
    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None)
    net = tf.squeeze(net, squeeze_dims=[1,2])
    logits_cls = tf.identity(net, 'logits')
    
    return logits_fcn, logits_cls, 16


def resnet_tiny2 (inputs, num_classes=2, scope ='resnet_tiny'):
    # seperate cls and fcn, add stop_gradient after fcn output
    blocks = [ 
        resnet_utils.Block('block1', resnet_v1.bottleneck,
                           [(64, 32, 1)] + [(64, 32, 2)]),
        resnet_utils.Block('block2', resnet_v1.bottleneck,
                           [(128, 64, 1)] + [(128, 64, 2)]),
        resnet_utils.Block('block3', resnet_v1.bottleneck,
                           [(256, 64, 1)] + [(128, 64, 2)]),
        resnet_utils.Block('block4', resnet_v1.bottleneck, [(128, 64, 1)])
        ]   
    net,_ = resnet_v1.resnet_v1(
        inputs, blocks,
        # all parameters below can be passed to resnet_v1.resnet_v1_??
        num_classes = None,       # don't produce final prediction
        global_pool = False,       # produce 1x1 output, equivalent to input of a FC layer
        output_stride = 16,
        include_root_block=True,
        reuse=False,              # do not re-use network
        scope=scope)
    res_out = net         # keep this for later CLS usage
    net = slim.batch_norm(slim.conv2d_transpose(net, 64, 5, 2))
    net = slim.batch_norm(slim.conv2d_transpose(net, 32, 5, 2))
    net = slim.batch_norm(slim.conv2d_transpose(net, 16, 5, 2))
    net = slim.batch_norm(slim.conv2d_transpose(net, 8, 5, 2))
    net = slim.conv2d(net, num_classes, 5, 1, activation_fn=None) 
    logits_fcn = tf.identity(net, 'logits_fcn')
    
    net = res_out
    net = tf.stop_gradient(net)
    # add a few layers to make the image size even smaller
    net = slim.conv2d(net, 128, 3, 1)
    net = slim.max_pool2d(net, 2, 2)
    net = slim.conv2d(net, 128, 3, 1)
    net = slim.max_pool2d(net, 2, 2)
    net = tf.reduce_mean(net, [1, 2], keep_dims=True)
    # add an extra layer
    net = slim.conv2d(net, 64, [1, 1])
    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None)
    logits_cls = tf.identity(net, 'logits_cls')
    
    return logits_fcn, logits_cls, 16
