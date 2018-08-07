import glob
import libs.preprocess.voc as preprocess
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow as tf

from config import *
from libs.datasets.VOC12 import VOC_CATS

def get_dataset(tfrecord_path):
  keys_to_features = {
    'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
    'image/format': tf.FixedLenFeature((), tf.string, default_value='JPEG'),
    'image/segmentation/encoded': tf.FixedLenFeature((), tf.string,default_value=''),
    'image/segmentation/format': tf.FixedLenFeature((), tf.string,default_value='RAW')
    }

  items_to_handlers = {
    'image': slim.tfexample_decoder.Image('image/encoded', 'image/format', channels=3),
    'segmentation': slim.tfexample_decoder.Image('image/segmentation/encoded', 'image/segmentation/format', channels=1),
    }

  items_to_descriptions = {
    'image': 'A color image of varying height and width.',
    'segmentation': 'A semantic segmentation.',
    }

  categories = None
  num_samples = 0
  if 'IDImage' in tfrecord_path:
    categories = ['__background__','id_card']
    num_samples = 515
  if 'VOC12' in tfrecord_path:
    categories = VOC_CATS
    num_samples = 14270

  return slim.dataset.Dataset(
    data_sources=[tfrecord_path],
    reader=tf.TFRecordReader,
    decoder=slim.tfexample_decoder.TFExampleDecoder(keys_to_features,items_to_handlers),
    num_samples=num_samples,
    items_to_descriptions=items_to_descriptions,
    num_classes=len(categories),
    labels_to_names={i: cat for i, cat in enumerate(categories)})

def extract_batch(dataset, batch_size, is_training):
  with tf.device("/cpu:0"):
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
      dataset, num_readers=4, shuffle=False, common_queue_capacity=128, common_queue_min=32) #原来是512

    image, gt_mask = data_provider.get(['image', 'segmentation'])
    image, gt_mask = preprocess.preprocess_image(image, gt_mask, is_training=is_training)
 
    return tf.train.shuffle_batch([image, gt_mask], batch_size, 256, 64, num_threads=4) # 原来是4096

def read_data(is_training, split_name):
  file_pattern = '{}_{}.tfrecord'.format(args.data_name, split_name)
  tfrecord_path = os.path.join(args.data_dir,'records',file_pattern)

  if is_training:
    dataset = get_dataset(tfrecord_path)
    image, gt_mask = extract_batch(dataset, args.batch_size, is_training)
  else:
    image, gt_mask = read_tfrecord(tfrecord_path)
    image, gt_mask = preprocess.preprocess_image(image, gt_mask, is_training)
  return image, gt_mask

def read_tfrecord(tfrecords_filename):
  if not isinstance(tfrecords_filename, list):
    tfrecords_filename = [tfrecords_filename]
  filename_queue = tf.train.string_input_producer(tfrecords_filename)

  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(serialized_example,features={'image/encoded': tf.FixedLenFeature([], tf.string),'image/segmentation/encoded': tf.FixedLenFeature([], tf.string),})
  image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
  gt_mask = tf.image.decode_png(features['image/segmentation/encoded'], channels=1)
  
  return image, gt_mask
