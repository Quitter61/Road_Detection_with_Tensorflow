#!/usr/bin/python
# -*- coding:utf-8 -*-

from __future__ import division

import skimage.io as io
import numpy as np
import tensorflow as tf
import pydensecrf.densecrf as dcrf
import sys
import os
from matplotlib import pyplot as plt
from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian, softmax_to_unary


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

  channels = tf.split(2, num_channels, image)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(2, channels)


##################main##################

checkpoint_path = '/home/damon/PycharmProjects/python/Road_Detection/checkpoint/fcn32s_model5e.ckpt'

slim = tf.contrib.slim
number_of_classes = 2
upsample_factor = 32

image_filename = 'after_rain00039.jpeg'
image_filename_placeholder = tf.placeholder(tf.string)
feed_image_dict = {image_filename_placeholder: image_filename}
image = tf.read_file(image_filename_placeholder)
image_tensor = tf.image.decode_jpeg(contents=image, channels=3)

#image_tensor = tf.image.decode_png(contents=image, channels=3)
image_float = tf.to_float(image_tensor)
mean_image = _mean_image_subtraction(image_float, [_R_MEAN, _G_MEAN, _B_MEAN])
processed_image = tf.expand_dims(input=mean_image, axis=0)

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

    upsample_logits_shape = tf.pack([downsample_logits_shape[0],
                                     downsample_logits_shape[1] * upsample_factor,
                                     downsample_logits_shape[2] * upsample_factor,
                                     downsample_logits_shape[3]
                                     ])
    upsample_logits = tf.nn.conv2d_transpose(value=logits, filter=upsample_filter_tensor,
                                             output_shape=upsample_logits_shape,
                                             strides=[1, upsample_factor, upsample_factor, 1])


pred = tf.argmax(upsample_logits, dimension=3)
probabilities = tf.nn.softmax(upsample_logits)

initializer = tf.local_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(initializer)

    saver.restore(sess, "/home/damon/PycharmProjects/python/Road_Detection/checkpoint/fcn32s_model10e.ckpt")

    image_np, pred_np, final_probabilities = sess.run([image_tensor, pred, probabilities], feed_dict=feed_image_dict)

    io.imshow(image_np)
    io.show()

    plt.imshow(pred_np.squeeze())
    plt.show()
    #io.imshow(pred_np.squeeze())
    #io.show()


image = image_np

softmax = final_probabilities.squeeze()
#processed_probabilities = tf.squeeze(probabilities)
softmax = softmax.transpose((2, 0, 1))
#softmax = tf.transpose(processed_probabilities, (2, 0, 1))

# 输入数据应为概率值的负对数
# 你可以在softmax_to_unary函数的定义中找到更多信息
unary = softmax_to_unary(softmax)

# 输入数据应为C-连续的——我们使用了Cython封装器
unary = np.ascontiguousarray(unary)

d = dcrf.DenseCRF(image.shape[0] * image.shape[1], 2)

d.setUnaryEnergy(unary)

# 潜在地对空间上相邻的小块分割区域进行惩罚——促使产生更多空间连续的分割区域
#sdim 是每一维的缩放因子 shape是crf的形状
feats = create_pairwise_gaussian(sdims=(10, 10), shape=image.shape[:2])

d.addPairwiseEnergy(feats, compat=3,
                    kernel=dcrf.DIAG_KERNEL,
                    normalization=dcrf.NORMALIZE_SYMMETRIC)

# 这将创建与颜色相关的图像特征——因为我们从卷积神经网络中得到的分割结果非常粗糙，
# 我们可以使用局部的颜色特征来改善分割结果
#schan 每个通道的缩放因子，chdim 代表rgb 3通道
feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20),
                                   img=image, chdim=2)

d.addPairwiseEnergy(feats, compat=10,
                     kernel=dcrf.DIAG_KERNEL,
                     normalization=dcrf.NORMALIZE_SYMMETRIC)

Q = d.inference(5)

res = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))
#res = np.argmax(d.inference(5), axis=0).reshape((image.shape[0], image.shape[1]))
cmap = plt.get_cmap('bwr')

f, ax1 = plt.subplots(1, 1, sharey=True)
ax1.imshow(res, vmax=1.5, vmin=-0.4, cmap=cmap)
ax1.set_title('Segmentation with CRF post-processing')
plt.show()

