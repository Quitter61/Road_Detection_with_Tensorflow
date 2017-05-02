#!/usr/bin/python
# -*- coding:utf-8 -*-
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np
import skimage.io as io
import os, sys
import time
from matplotlib import pyplot as plt


########################################
################上采样##################
########################################

def get_kernel_size(factor):
    # 给定所需的上采样因子，确定卷积核的大小
    return 2 * factor - factor % 2


def upsample_filt(size):
    # 创建一个给定(h, w)大小的适用于上采样过程的二维双线性卷积核
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)


def bilinear_upsample_weights(factor, number_of_classes):
    # 使用双线性卷积核，为转置卷积创建权重矩阵并初始化.
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
################定义vgg16网络##############
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
            # 使用 conv2d 代替全連接層.
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
#################图像减均值##################
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


############################################
######图片的输入和解码，最好变为一维########
############################################

def read_and_decode(tfrecord_filenames_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(tfrecord_filenames_queue)

    features = tf.parse_single_example(serialized_example,
                                       features={'height': tf.FixedLenFeature([], tf.int64),
                                                 'width': tf.FixedLenFeature([], tf.int64),
                                                 'mask_raw': tf.FixedLenFeature([], tf.string),
                                                 'image_raw': tf.FixedLenFeature([], tf.string)})

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    gt_image = tf.decode_raw(features['mask_raw'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    image_shape = tf.pack([height, width, 3])
    gt_image_shape = tf.pack([height, width, 1])

    image = tf.reshape(image, image_shape)
    gt_image = tf.reshape(gt_image, gt_image_shape)

    resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                           target_height=480,
                                                           target_width=640)
    resized_gt_image = tf.image.resize_image_with_crop_or_pad(image=gt_image,
                                                              target_height=480,
                                                              target_width=640)


    image, gt_image = tf.train.shuffle_batch([resized_image, resized_gt_image],
                                             batch_size=1,
                                             capacity=3000,
                                             num_threads=2,
                                             min_after_dequeue=1000)

    return image, gt_image

#####################################
#############去除fc8層權重#############
#####################################
def extract_vgg_16_mapping_without_fc8(vgg_16_variables_mapping):
    vgg_16_keys = vgg_16_variables_mapping.keys()
    vgg_16_without_fc8_keys = []
    for key in vgg_16_keys:
        if 'fc8' not in key:
            vgg_16_without_fc8_keys.append(key)
    updated_mapping = {key: vgg_16_variables_mapping[key] for key in vgg_16_without_fc8_keys}
    return updated_mapping

######################程序开始##########################
# 選擇使用哪一個GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# 添加庫的路徑
# sys.path.append(r"D:\python\FCNs_ImageSegmentation\tf-image-segmentation")

checkpoints_dir = '/home/damon/PycharmProjects/python/second_dataset/32s'
log_folder = '/home/damon/PycharmProjects/python/second_dataset/32s/log_folder'

slim = tf.contrib.slim
vgg_checkpoint_path = os.path.join(checkpoints_dir, 'vgg_16.ckpt')

fig_size = [16, 4]
plt.rcParams["figure.figsize"] = fig_size

upsample_factor = 32
number_of_classes = 2
tfrecord_filename = 'second_dataset_train.tfrecords'

tfrecord_filenames_queue = tf.train.string_input_producer([tfrecord_filename])

image, gt_image = read_and_decode(tfrecord_filenames_queue)

'''
###gt_image不进行取等处理
gt_image_float = tf.to_float(gt_image)
flat_labels = tf.reshape(tensor=gt_image_float, shape=(-1, 2))
'''
###对gt_image进行类别和flatter处理###
class_labels_tensor = tf.equal(gt_image, 255)
background_labels_tensor = tf.not_equal(gt_image, 255)

class_labels_tensor = tf.to_float(class_labels_tensor)
background_labels_tensor = tf.to_float(background_labels_tensor)
class_labels_tensor = tf.squeeze(class_labels_tensor, axis=0)
background_labels_tensor = tf.squeeze(background_labels_tensor, axis=0)

combined_labels = tf.concat(concat_dim=2, values=[class_labels_tensor, background_labels_tensor])
flat_labels = tf.reshape(tensor=combined_labels, shape=(-1, 2))

###对image进行相应的处理
image_float = tf.to_float(image)
image_float = tf.squeeze(image_float)

mean_image_tensor = _mean_image_subtraction(image=image_float, means=[_R_MEAN, _G_MEAN, _B_MEAN])

processed_images = tf.expand_dims(mean_image_tensor, 0)

upsample_filter_np = bilinear_upsample_weights(upsample_factor, number_of_classes)
upsample_filter_tensor = tf.constant(upsample_filter_np)


with tf.variable_scope("fcn_32s") as fcn_32s_scope:
    with slim.arg_scope(vgg_arg_scope()):
        logits, end_points = vgg_16(processed_images,
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

    vgg_16_variables_mapping = {}
    vgg_16_variables = slim.get_variables(fcn_32s_scope)
    for variable in vgg_16_variables:
        original_vgg_16_checkpoint_string = variable.name[len(fcn_32s_scope.original_name_scope): -2]
        vgg_16_variables_mapping[original_vgg_16_checkpoint_string] = variable

flat_logits = tf.reshape(tensor=upsample_logits, shape=(-1, number_of_classes))

# 计算logits和labels的交叉熵
cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                          labels=flat_labels)

# 平均交叉熵
cross_entropy_sum = tf.reduce_mean(cross_entropies)

# upsample_logist_batch,在第三维上最大值的索引
pred = tf.argmax(upsample_logits, dimension=3)
probabilities = tf.nn.softmax(upsample_logits)

with tf.variable_scope("adam_vars"):
    train_step = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(cross_entropy_sum)

# Variable's initialization functions
# vgg_16_without_fc8_variables_mapping = extract_vgg_16_mapping_without_fc8(vgg_16_variables_mapping)
#vgg_except_fc8_weights = slim.get_variables_to_restore(exclude=['vgg_16/fc8', 'adam_vars'])
#vgg_fc8_weights = slim.get_variables_to_restore(include=['vgg_16/fc8'])
#vgg_fc8_weights_initializer = tf.variables_initializer(vgg_fc8_weights)

vgg_16_without_fc8_variables_mapping = extract_vgg_16_mapping_without_fc8(vgg_16_variables_mapping)

init_fn = slim.assign_from_checkpoint_fn(model_path=vgg_checkpoint_path,
                                         var_list=vgg_16_without_fc8_variables_mapping)

global_vars_init_op = tf.global_variables_initializer()

tf.summary.scalar('cross_entropy_loss', cross_entropy_sum)

merged_summary_op = tf.summary.merge_all()

summary_string_writer = tf.summary.FileWriter(log_folder)

# 創建log_folder 文件夾
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

# The op for initializing the variables.
local_vars_init_op = tf.local_variables_initializer()

combined_op = tf.group(local_vars_init_op, global_vars_init_op)

# 保存權重參數
model_variables = slim.get_model_variables()
saver = tf.train.Saver(model_variables)

cross_entropy_plt = []

# config = tf.ConfigProto()
# config.gpu_options.allocator_type = 'BFC'
start_time = time.time()
with tf.Session() as sess:
    sess.run(combined_op)
    #sess.run(vgg_fc8_weights_initializer)
    init_fn(sess)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # 529 epochs
    for i in range(529*10):

        cross_entropy, summary_string, _ = sess.run([cross_entropy_sum,
                                                     merged_summary_op,
                                                     train_step])
        pred_np, probabilities_np = sess.run([pred, probabilities])

        print("Iteration: " + str(i), "Current loss: " + str(cross_entropy))
        cross_entropy_plt.append(cross_entropy)

        summary_string_writer.add_summary(summary_string, i)
        '''
        cmap = plt.get_cmap('bwr')

        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.imshow(np.uint8(pred_np.squeeze() != 1), vmax=1.5, vmin=-0.4, cmap=cmap)
        ax1.set_title('Argmax. Iteration # ' + str(i))
        probability_graph = ax2.imshow(probabilities_np.squeeze()[:, :, 0])
        ax2.set_title('Probability of the Class. Iteration # ' + str(i))

        plt.colorbar(probability_graph)
        plt.show()
        '''
        if i % 529 == 0:
            save_path = saver.save(sess,"/home/damon/PycharmProjects/python/second_dataset/32s/checkpoint/fcn32s_model10e_2.ckpt")
            print("Model saved in file: %s" % save_path)

    coord.request_stop()
    coord.join(threads)

    save_path = saver.save(sess, "/home/damon/PycharmProjects/python/second_dataset/32s/checkpoint/fcn32s_model10e_2.ckpt")
    print("Model saved in file: %s" % save_path)
print("train_time", time.time() - start_time)
summary_string_writer.close()

plt.figure(figsize=(50, 6), dpi=300)
plt.subplot(1, 1, 1)
X = np.linspace(start=0, stop=529*10, num=529*10, endpoint=False)
Y = cross_entropy_plt

plt.plot(X, Y, color="blue", linewidth=1, linestyle='-', label="BCN-1 Loss")
plt.ylim(0, 1)
plt.legend(loc='upper right')

plt.show()