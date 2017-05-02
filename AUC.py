#!/usr/bin/python
# -*- coding:utf-8 -*-

import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt

####defind read and decode function####
def read_and_decode(tfrecord_filenames_queue):

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(tfrecord_filenames_queue)

    features = tf.parse_single_example(serialized_example,
                                       features={'height': tf.FixedLenFeature([], tf.int64),
                                                 'width': tf.FixedLenFeature([], tf.int64),
                                                 'mask_raw': tf.FixedLenFeature([], tf.string),
                                                 'image_raw': tf.FixedLenFeature([], tf.string) })

    gt_image = tf.decode_raw(features['mask_raw'], tf.uint8)
    pred_image = tf.decode_raw(features['image_raw'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    gt_image_shape = tf.pack([height, width, 1])
    pred_image_shape = tf.pack([height, width, 1])

    gt_image = tf.reshape(gt_image, gt_image_shape)
    pred_image = tf.reshape(pred_image, pred_image_shape)

    return gt_image, pred_image

############ main #################
slim = tf.contrib.slim

tfrecord_filename = 'fcn8s_10e_crf_2.tfrecords'
tfrecord_filenames_queue = tf.train.string_input_producer([tfrecord_filename])
gt_image, pred_image = read_and_decode(tfrecord_filenames_queue)

gt_image_float = tf.to_float(gt_image)
pred_image_float = tf.to_float(pred_image)

gt_image_tensor = tf.expand_dims(gt_image_float, 0)
pred_image_tensor = tf.expand_dims(pred_image_float, 0)

'''
miou, update_op = slim.metrics.streaming_mean_iou(predictions=pred_image_tensor,
                                                  labels=gt_image_tensor,
                                                  num_classes=2)
'''
auc, update_op = tf.contrib.metrics.streaming_auc(predictions=pred_image_tensor,
                                                  labels=gt_image_tensor,
                                                  num_thresholds=10)

initializer = tf.local_variables_initializer()
#initializer = tf.global_variables_initializer()

start_time = time.time()
with tf.Session() as sess:
    sess.run(initializer)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # There are 226 images
    for i in range(226):
        gt_image_np, pred_image_np, pred_np, tmp = sess.run([gt_image, pred_image, pred_image_tensor, update_op])

        print("current auc: ", tmp)
        #pred_img = pred_np.squeeze()
        #io.imsave('pred.png', pred_img)
        #plt.imshow(pred_np.squeeze())
        #plt.show()
        # Display the image and the segmentation result
        # upsampled_predictions = pred_np.squeeze()
        # plt.imshow(gt_image_np.squeeze())
        # plt.show()
        # plt.imshow(pred_image_np.squeeze())
        # plt.show()
        # visualize_segmentation_adaptive(upsampled_predictions, pascal_voc_lut)
    coord.request_stop()
    coord.join(threads)

    ROC = sess.run(auc)

    print("Mean auc: " + str(ROC))

print("total time: ", time.time()-start_time)