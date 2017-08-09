#!/usr/bin/env python
# encoding: utf-8

from __future__ import division

import numpy as np
import tensorflow as tf
import utils
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import time
import os
###########  hyper parameters ###########
slim = tf.contrib.slim
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
number_of_classes = 2
upsample_factor = 8
class seger(object):
    def __init__(self):
        self.image=tf.placeholder(dtype=tf.float32, shape=[514,514,3])
        self.session,self.pred,self.probabilities=self.seg()

    def seg(self):
        image = self.image
        #image_tensor = tf.stack(values=image)
        image_float = tf.to_float(image)
        mean_image = utils.mean_image_subtraction(image_float, [_R_MEAN, _G_MEAN, _B_MEAN])
        processed_image = tf.expand_dims(input=mean_image, axis=0)

        upsample_filter_8s_np = utils.bilinear_upsample_weights(upsample_factor, number_of_classes)
        upsample_filter_2s_np = utils.bilinear_upsample_weights(2, number_of_classes)

        upsample_filter_8_tensor = tf.constant(upsample_filter_8s_np)
        upsample_filter_2_tensor = tf.constant(upsample_filter_2s_np)

        with tf.variable_scope("fcn_8s")  as fcn_8s_scope:
            with slim.arg_scope(utils.vgg_arg_scope()):
                last_layer_logits, end_points = utils.vgg_16(processed_image,
                                                             num_classes=2,
                                                             is_training=False,
                                                             spatial_squeeze=False,
                                                             fc_conv_padding='SAME')

            last_layer_logits_shape = tf.shape(last_layer_logits)
            # last downsample layer

            last_layer_upsample_logits_shape = tf.stack([last_layer_logits_shape[0],
                                                         last_layer_logits_shape[1] * 2,
                                                         last_layer_logits_shape[2] * 2,
                                                         last_layer_logits_shape[3]])

            last_layer_upsample_logits = tf.nn.conv2d_transpose(value=last_layer_logits,
                                                                filter=upsample_filter_2_tensor,
                                                                output_shape=last_layer_upsample_logits_shape,
                                                                strides=[1, 2, 2, 1])

            pool4_features = end_points['fcn_8s/vgg_16/pool4']
            pool4_logits = slim.conv2d(pool4_features, number_of_classes, [1, 1], activation_fn=None,
                                       normalizer_fn=None, weights_initializer=tf.zeros_initializer,
                                       scope='pool4_fc')
            fused_last_layer_and_pool4_logits = pool4_logits + last_layer_upsample_logits
            fused_last_layer_and_pool4_logits_shape = tf.shape(fused_last_layer_and_pool4_logits)

            fused_last_layer_and_pool4_upsampled_by_factor_2_logits_shape = tf.stack(
                [fused_last_layer_and_pool4_logits_shape[0],
                 fused_last_layer_and_pool4_logits_shape[1] * 2,
                 fused_last_layer_and_pool4_logits_shape[2] * 2,
                 fused_last_layer_and_pool4_logits_shape[3]])
            fused_last_layer_and_pool4_upsampled_by_factor_2_logits = tf.nn.conv2d_transpose(
                value=fused_last_layer_and_pool4_logits,
                filter=upsample_filter_2_tensor,
                output_shape=fused_last_layer_and_pool4_upsampled_by_factor_2_logits_shape,
                strides=[1, 2, 2, 1])

            pool3_features = end_points['fcn_8s/vgg_16/pool3']

            pool3_logits = slim.conv2d(pool3_features, number_of_classes, [1, 1], activation_fn=None,
                                       normalizer_fn=None, weights_initializer=tf.zeros_initializer,
                                       scope='pool3_fc')

            fused_last_layer_and_pool4_and_pool3_logits = pool3_logits + fused_last_layer_and_pool4_upsampled_by_factor_2_logits

            fused_last_layer_and_pool4_and_pool3_logits_shape = tf.shape(fused_last_layer_and_pool4_and_pool3_logits)

            fused_last_layer_pool4_pool3_upsampled_by_8_logits_shape = tf.stack(
                [fused_last_layer_and_pool4_and_pool3_logits_shape[0],
                 fused_last_layer_and_pool4_and_pool3_logits_shape[1] * upsample_factor,
                 fused_last_layer_and_pool4_and_pool3_logits_shape[2] * upsample_factor,
                 fused_last_layer_and_pool4_and_pool3_logits_shape[3]])
            fused_last_layer_pool4_pool3_logits = tf.nn.conv2d_transpose(
                value=fused_last_layer_and_pool4_and_pool3_logits,
                filter=upsample_filter_8_tensor,
                output_shape=fused_last_layer_pool4_pool3_upsampled_by_8_logits_shape,
                strides=[1, upsample_factor, upsample_factor, 1])

        pred = tf.argmax(fused_last_layer_pool4_pool3_logits, dimension=3)
        probabilities = tf.nn.softmax(fused_last_layer_pool4_pool3_logits)

        initializer = tf.local_variables_initializer()
        saver = tf.train.Saver()
        _gpu_options = tf.GPUOptions(allow_growth=False,
                                 per_process_gpu_memory_fraction=0.2,
                                 visible_device_list='0')

        if not os.environ.get('OMP_NUM_THREADS'):
            config = tf.ConfigProto(allow_soft_placement=True,
                                gpu_options=_gpu_options)
        else:
            num_thread = int(os.environ.get('OMP_NUM_THREADS'))
            config = tf.ConfigProto(intra_op_parallelism_threads=num_thread,
                                allow_soft_placement=True,
                                gpu_options=_gpu_options)
        sess = tf.Session(config=config)
        sess.run(initializer)
        # 模型路径
        saver.restore(sess, r"./seg_model/checkpoint_8s_6k/fcn8s_10e.ckpt")
        return sess,pred,probabilities

    def session_run(self, image_):
        pred_np,final_probabilities = self.session.run([self.pred,self.probabilities], feed_dict={self.image:image_})
        return (pred_np.squeeze())
