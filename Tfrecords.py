#!/usr/bin/python
# -*- coding:utf-8 -*-

from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.io as io

'''
#定义图片读取路径
filename_pairs = [
('/home/damon/Second_road_image/road_image_dataset/input_003.png', '/home/damon/Second_road_image/road_image_dataset/input_003_GT.png'),
('/home/damon/Second_road_image/road_image_dataset/input_005.png', '/home/damon/Second_road_image/road_image_dataset/input_005_GT.png'),
('/home/damon/Second_road_image/road_image_dataset/input_007.png', '/home/damon/Second_road_image/road_image_dataset/input_007_GT.png'),
('/home/damon/Second_road_image/road_image_dataset/input_011.png', '/home/damon/Second_road_image/road_image_dataset/input_011_GT.png'),
]
'''

def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

file_img = get_imlist(r'')
file_gt = get_imlist(r'')


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

tfrecords_filename = 'second_dataset_test.tfrecords'

#写入数据
write = tf.python_io.TFRecordWriter(tfrecords_filename)

original_images = []

for img_path, gt_img_path in zip(file_img, file_gt):
    img = np.array(Image.open(img_path))
    gt_img = np.array(Image.open(gt_img_path))

    # plt.imshow(gt_img)
    # plt.show()
    # plt.imshow(img)
    # plt.show()
    height = img.shape[0]
    width = img.shape[1]

    original_images.append((img, gt_img))

    #图片变为字符型
    img_raw = img.tostring()
    gt_img_raw = gt_img.tostring()

    #定义数据格式
    example = tf.train.Example(features=tf.train.Features(feature={'height': _int64_feature(height),
                                                                  'width': _int64_feature(width),
                                                                   'mask_raw': _bytes_feature(gt_img_raw),
                                                                  'image_raw': _bytes_feature(img_raw)}))

    write.write(example.SerializeToString())
write.close()

reconstructed_images = []
record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

for string_record in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(string_record)

    height = int(example.features.feature['height'].int64_list.value[0])

    width = int(example.features.feature['width'].int64_list.value[0])

    img_string = (example.features.feature['image_raw'].bytes_list.value[0])
    gt_img_string = (example.features.feature['mask_raw'].bytes_list.value[0])

    img_1d = np.fromstring(img_string, dtype=np.uint8)
    reconstructed_img = img_1d.reshape((height, width, -1))
    # plt.imshow(reconstructed_img)
    # plt.show()

    gt_img_1d = np.fromstring(gt_img_string, dtype=np.uint8)
    reconstructed_gt_img = gt_img_1d.reshape((height, width))
    #plt.imshow(reconstructed_gt_img)
    #plt.show()


    reconstructed_images.append((reconstructed_img, reconstructed_gt_img))

for original_pair, reconstructed_pair in zip(original_images, reconstructed_images):
    img_pair_to_compare, gt_img_pair_to_compare = zip(original_pair,
                                                          reconstructed_pair)
    print(np.allclose(*img_pair_to_compare))
    print(np.allclose(*gt_img_pair_to_compare))


