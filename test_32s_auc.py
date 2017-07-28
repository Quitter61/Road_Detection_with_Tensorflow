#!/usr/bin/env python
# encoding: utf-8

from __future__ import division

import skimage.io as io
import numpy as np
import tensorflow as tf
import sys
import os
from matplotlib import pyplot as plt


##################################
##########upsample################
##################################

def get_kernel_size(factor):
    """
    Find the kernel size given the desired factor of upsampling.
    """
    return 2 * factor - factor % 2


def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def bilinear_upsample_weights(factor, number_of_classes):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """

    filter_size = get_kernel_size(factor)

    weights = np.zeros((filter_size,
                        filter_size,
                        number_of_classes,
                        number_of_classes), dtype=np.float32)

    upsample_kernel = upsample_filt(filter_size)

    for i in range(number_of_classes):
        weights[:, :, i, i] = upsample_kernel

    return weights

###########################################
################net vgg16##############
###########################################

###定义vgg_arg_scope
def vgg_arg_scope(weight_decay=0.0005):

  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc

def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='SAME'):

  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')
      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout7')
      net = slim.conv2d(net, num_classes, [1, 1],
                        activation_fn=None,
                        normalizer_fn=None,
                        scope='fc8')
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if spatial_squeeze:
        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points
vgg_16.default_image_size = 224


#############################################
##########mean image subtraction#############
#############################################
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

def _mean_image_subtraction(image, means):

  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(value=image, num_or_size_splits=num_channels, axis=2)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(values=channels, axis=2)

############################################
##########read and decode images############
############################################

def read_and_decode(tfrecord_filenames_queue):

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(tfrecord_filenames_queue)

    features = tf.parse_single_example(serialized_example,
                                       features={'height': tf.FixedLenFeature([], tf.int64),
                                                 'width': tf.FixedLenFeature([], tf.int64),
                                                 'mask_raw': tf.FixedLenFeature([], tf.string),
                                                 'image_raw': tf.FixedLenFeature([], tf.string) })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    gt_image = tf.decode_raw(features['mask_raw'], tf.uint8)

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    image_shape = tf.stack([height, width, 3])
    gt_image_shape = tf.stack([height, width, 1])

    image = tf.reshape(image, image_shape)
    gt_image = tf.reshape(gt_image, gt_image_shape)


    return image, gt_image


##################main##################

# checkpoint_path = r"E:\python program\model_32s_5e\checkpoint\fcn32s_5e.ckpt"

slim = tf.contrib.slim
number_of_classes = 2
upsample_factor = 32

tfrecord_filename = 'train_data_5w.tfrecords'
tfrecord_filenames_queue = tf.train.string_input_producer([tfrecord_filename])
image, gt_image = read_and_decode(tfrecord_filenames_queue)

#预处理
gt_image_labels = tf.equal(gt_image, 0)      #logits的目标（0）背景（1），此处将背景变为1，目标变为0
gt_image_float = tf.to_float(gt_image_labels)
image_float = tf.to_float(image)

mean_image_tensor = _mean_image_subtraction(image=image_float, means=[_R_MEAN, _G_MEAN, _B_MEAN])
processed_image = tf.expand_dims(mean_image_tensor, 0)
processed_gt_image = tf.expand_dims(gt_image_float, 0)

upsample_filter_np = bilinear_upsample_weights(upsample_factor, number_of_classes)
upsample_filter_tensor = tf.constant(upsample_filter_np)

with tf.variable_scope("fcn_32s") as fcn_32s_scope:
    with slim.arg_scope(vgg_arg_scope()):

        logits, end_points = vgg_16(processed_image,
                                    num_classes=2,
                                    is_training=False,
                                    spatial_squeeze=False,
                                    fc_conv_padding='SAME')

    downsample_logits_shape = tf.shape(logits)

    upsample_logits_shape = tf.stack([downsample_logits_shape[0],
                                     downsample_logits_shape[1] * upsample_factor,
                                     downsample_logits_shape[2] * upsample_factor,
                                     downsample_logits_shape[3]
                                     ])
    upsample_logits = tf.nn.conv2d_transpose(value=logits, filter=upsample_filter_tensor,
                                             output_shape=upsample_logits_shape,
                                             strides=[1, upsample_factor, upsample_factor, 1])


pred = tf.argmax(upsample_logits, dimension=3)
probabilities = tf.nn.softmax(upsample_logits)

auc, update_op = tf.contrib.metrics.streaming_auc(predictions=pred,
                                                  labels=processed_gt_image,
                                                  num_thresholds=10)

initializer = tf.local_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(initializer)

    # saver.restore(sess, r"E:\python program\checkpoint_ing\fcn32s_10e.ckpt")
    saver.restore(sess, r"E:\python program\checkpoint_ing4\fcn32s_10e.ckpt")

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(100):

        image_np, gt_image_np, pred_np, tmp = sess.run([image, gt_image, pred, update_op])

        coord.request_stop()
        coord.join(threads)

        res = sess.run(auc)

        io.imshow(image_np)
        io.show()

        plt.imshow(gt_image_np.squeeze())
        plt.show()

        plt.imshow(pred_np.squeeze())
        plt.show()

        print("AUC: " + str(res))

