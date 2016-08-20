# Downloading CIFAR 10:                                                                                                                                                               
import os
import six
import sys
from six.moves import urllib
import tarfile

# Create a TFRecord Dataset                                                                                                                                                           
import cPickle
import numpy as np
import os
import sys
import tensorflow as tf

def download_data(folder):
    '''
    The zip file contains the following files:
    batches.meta  data_batch_2  data_batch_4  readme.html
    data_batch_1  data_batch_3  data_batch_5  test_batch
    '''
    DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    if not os.path.exists(folder):
        os.makedirs(folderr)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(folder, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(folder)

### Code to convert raw data to tfrecord format

def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def _encode_image(image):
    with tf.Graph().as_default():
        with tf.Session(''):
            return tf.image.encode_png(tf.constant(image)).eval()

def _add_to_tfrecord(filename, tfrecord_writer, offset=0):
    with tf.gfile.Open(filename, 'r') as f:
        data = cPickle.load(f)

    images = data['data']
    num_images = images.shape[0]

    images = images.reshape((num_images, 3, 32, 32))
    labels = data['labels']

    for j in range(num_images):
        sys.stdout.write('\r>> Reading image from file %s (%d/%d)' % (filename, offset + j + 1, offset + num_images))
        sys.stdout.flush()

        image = np.squeeze(images[j]).transpose((1, 2, 0))
        label = labels[j]

        png_string = _encode_image(image)
    
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/encoded': _bytes_feature(png_string),
            'image/format': _bytes_feature('png'),
            'image/class/label': _int64_feature(label),
            'image/height': _int64_feature(32),
            'image/width': _int64_feature(32),
        }))

        tfrecord_writer.write(example.SerializeToString())

    return offset + num_images

def get_tf_filename(split_name, folder):
    return '%s/cifar10_%s.tfrecord' % (folder, split_name)

def convert_to_tfrecord(raw_folder, tf_folder):
    '''
    We create 2 tfrecord files, one for train and one for test.
    '''
    NUM_TRAIN_FILES = 5
    if not os.path.exists(tf_folder):
        os.makedirs(tf_folder)
    input_dir = raw_folder + 'cifar-10-batches-py/'
    output_file = get_tf_filename('train', tf_folder)
    with tf.python_io.TFRecordWriter(output_file) as tfrecord_writer:
        # Create training tf record file.
        offset = 0
        for i in range(NUM_TRAIN_FILES):
            filename = os.path.join(input_dir, 'data_batch_%d' % (i + 1))  # 1-indexed.
            offset = _add_to_tfrecord(filename, tfrecord_writer, offset)
        # Create testing tf record file.
        output_file = get_tf_filename('test', tf_folder)
        with tf.python_io.TFRecordWriter(output_file) as tfrecord_writer:
            filename = os.path.join(input_dir, 'test_batch')
            _add_to_tfrecord(filename, tfrecord_writer)
            

def main(unused_args):
    CIFAR10_RAW_DATA_DIR = '/tmp/cifar10_raw/'
    CIFAR10_DIR = '/tmp/cifar10_tf/'
    download_data(CIFAR10_RAW_DATA_DIR)
    convert_to_tfrecord(CIFAR10_RAW_DATA_DIR, CIFAR10_DIR)
