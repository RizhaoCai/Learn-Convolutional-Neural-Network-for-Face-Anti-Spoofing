import tensorflow as tf
import numpy as np
#import .load_data
def np2tf(X, Y):
    output_file = 'train.tfrecords'
    writer = tf.python_io.TFRecordWriter(output_file)
    example = tf.train.Example()

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
     byted_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
     return byted_feature


def np2tfrecord(X, Y, save_path='train.tfrecords'):

    writer = tf.python_io.TFRecordWriter( save_path)
    num = X.shape[0]
    size_X =X.shape[2]
    size_Y = Y.shape[1]
    
    for i in range(num):
        print(X[i].shape)
        example = tf.train.Example(
            features=tf.train.Features(feature={
                'face': _bytes_feature( X[i,:,:,:].tobytes()),
                'depth':  _bytes_feature(Y[i,:,:,:].tobytes()),
                #'size_X': size_X,
                #'size_Y': size_Y \
                    }))
        print('Writing instance: ',i)
        writer.write(example.SerializeToString())
    writer.close()

def read_tf_record(filepath='train.tfrecords'):
    reader = tf.TFRecordReader()
    file_queue=tf.train.string_input_producer([filepath])
    _, serialized_example = reader.read_up_to(file_queue,100)
    features=tf.parse_example(
        serialized_example,
        features={
           'face': tf.FixedLenFeature([], tf.string),
            'depth': tf.FixedLenFeature([], tf.string),
        }
    )
    images = tf.decode_raw(features['face'], tf.uint8)
    depth_map = tf.decode_raw(features['depth'], tf.uint8)
    print(tf.shape(images))
    print(tf.shape(depth_map))
    return images, depth_map


def make_dataset(X,Y):
    dataset = tf.data.Dataset.from_tensor_slices((X,Y))

if __name__ == '__main__':
    filepath = '/data/cairizhao/train.tfrecords'
    example = tf.train.Example()
    print('Loading raw data')
    #X,Y = load_data.load_data('/data/cairizhao/exp/C2RTT.mat','TRAIN')
    #np2tfrecord(X,Y,filepath)
    _,_i =read_tf_record(filepath)
    #import IPython; IPython.embed()
