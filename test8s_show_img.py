#!/usr/bin/env python
# encoding: utf-8

from __future__ import division

import numpy as np
import tensorflow as tf
import utils
import matplotlib.pyplot as plt

from PIL import Image

###########  hyper parameters ###########
slim = tf.contrib.slim
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
number_of_classes = 2
upsample_factor = 8

# image = Image.open('2_63.jpg')
# image = np.array(image)
# image = tf.stack(values=image,)

def seg(seg_img):

    # image_filename = r'E:\rawdata\test_slice\Tumor_001\1_64.jpg'
    # image_filename_placeholder = tf.placeholder(tf.string)
    # feed_image_dict = {image_filename_placeholder: image_filename}
    # image = tf.read_file(image_filename_placeholder)
    # image_tensor = tf.image.decode_jpeg(contents=image, channels=3)
    # image_tensor = tf.image.decode_png(contents=image, channels=3)
    image = seg_img
    image_tensor = tf.stack(values=image)
    image_float = tf.to_float(image_tensor)
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
        fused_last_layer_pool4_pool3_logits = tf.nn.conv2d_transpose(value=fused_last_layer_and_pool4_and_pool3_logits,
                                                                     filter=upsample_filter_8_tensor,
                                                                     output_shape=fused_last_layer_pool4_pool3_upsampled_by_8_logits_shape,
                                                                     strides=[1, upsample_factor, upsample_factor, 1])

    pred = tf.argmax(fused_last_layer_pool4_pool3_logits, dimension=3)
    probabilities = tf.nn.softmax(fused_last_layer_pool4_pool3_logits)

    initializer = tf.local_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(initializer)

        # saver.restore(sess, r"E:\python program\checkpoint_ing\fcn32s_10e.ckpt")
        saver.restore(sess, r"E:\python program\checkpoint_8s_6k\fcn8s_10e.ckpt")

        image_np, pred_np, final_probabilities = sess.run([image_tensor, pred, probabilities])

        # plt.imshow(pred_np.squeeze())
        # plt.show()
        return(pred_np.squeeze())

seg()




