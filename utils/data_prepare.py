"""
    Convert raw pixel data to TFRecords file format with example proto.s
    cairizhao@email.szu.edu.cn
"""
import tensorflow as tf
import numpy as np
import os
import sys
import h5py as h5
import numpy as np
FLAGS = None
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'

def np2tf(X, Y):
    output_file = 'train.tfrecords'
    writer = tf.python_io.TFRecordWriter(output_file)
    example = tf.train.Example()

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def np2tfrecord(X, Y, label, save_path='train.tfrecords'):
    
    num_samples = X.shape[0]
    size_X =X.shape[-2]
    size_Y = Y.shape[-2]
    channel_X = X.shape[-1]
    channel_Y = Y.shape[-1]
    print('Writing', save_path)
    with tf.python_io.TFRecordWriter(save_path) as writer:
        for i in range(num_samples):
            print("Transform: {} done".format(i))
            X_raw = X[i].tostring()
            Y_raw = Y[i].tostring()
            lbl = label[i]
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'X': _bytes_feature(X_raw ),
                    'Y':  _bytes_feature(Y_raw),
                    'size_X': _int64_feature(size_X),
                    'size_Y':_int64_feature( size_Y),
                    'channel_X': _int64_feature(channel_X),
                     'channel_Y': _int64_feature(channel_Y),
                    'label':_int64_feature(lbl)
                        }))
            print("All Transformed! Serializing!")
            writer.write(example.SerializeToString())
            print(" Serializing Finished!")

def decode(serialized_example):
    """Parses an image and label from the given `serialized_example`."""
    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'X': tf.FixedLenFeature([], tf.string),
          'Y': tf.FixedLenFeature([], tf.string),
          'size_X': tf.FixedLenFeature([], tf.int64),
          'size_Y':tf.FixedLenFeature([], tf.int64),
          'channel_X': tf.FixedLenFeature([], tf.int64),
           'channel_Y': tf.FixedLenFeature([], tf.int64),
           'label': tf.FixedLenFeature([], tf.int64),
      })
    face_image = tf.decode_raw(features['X'], tf.uint8)
    depth_map = tf.decode_raw(features['Y'], tf.uint8)
    # When converted to serialized data, the information of shape gets lost, so it should be recovered
    face_shape = (256*256*3)
    depth_map_shape = (32*32*1)
    #print("before decode: {}".format(face_image.shape) )
    face_image.set_shape(face_shape)
    face_image=tf.reshape(face_image,[256,256,3])
    depth_map.set_shape(depth_map_shape)
    depth_map=tf.reshape(depth_map,[32,32,1])
    #print("after decode: {}".format(face_image.shape) )
    # Convert label from a scalar uint8 tensor to an int32 scalar.
    #label = tf.cast(features['label'], tf.int32)
    label = tf.cast(features['label'],tf.int8)
    return face_image, depth_map, label


def inputs_iterator(train, batch_size, num_epochs, filepath):
    """Reads input data num_epoch times
        Args:
        train: Selects between the training (True) and validation (False) data.
        batch_size: Number of examples per returned batch.
        num_epochs: Number of times to read the input data, or 0/None to
        train forever.
     Returns:
        A tuple (images, labels), where:
        * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
        in the range [-0.5, 0.5].
        * labels is an int32 tensor with shape [batch_size] with the true label,
        a number in the range [0, mnist.NUM_CLASSES).
        This function creates a one_shot_iterator, meaning that it will only iterate
        over the dataset once. On the other hand there is no special initialization
    """
    if not num_epochs:
        num_epochs = None
    filename = filepath
    with tf.name_scope('input'):
        # the 'filename' could also be a list of filenames, which will be read in order
        dataset = tf.data.TFRecordDataset(filename)
        #dataset = tf.data.TFRecordDataset([filename])
        # The map transformation taks a function and applies it to every element 
        # of the dataset
        dataset = dataset.map(decode)

        # The shuffle transformation uses a finite-sized buffer to shuffle elelemts
        # in memory. The parameter is the number of elements in the buffer. For 
        # completely uniform shuffling, set the parameter to be the same as the 
        # number of elements in the  dataset.
        dataset = dataset.shuffle(1000 + 3 * batch_size)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
    print("Iterator get next")
    return iterator
     

def load_h5_data(mat, set_type, stride=1):
     X = mat[set_type+'_X'][::stride]
     depth_map = mat[set_type+'_D'][::stride]
     label =  mat[set_type+'_LBL'][::stride]   
     #X = 1
     #depth_map=2
     
     return np.moveaxis(X,[1,2,3],[-1,-2,-3]), np.moveaxis(depth_map,[1,2,3],[-1,-2,-3]), label
     
     
    

if __name__ == '__main__':
    
    """
        TRAIN_X, TRAIN_D, TRAIN_LBL
        TEST_X, TEST_D, TEST_LBL
    """
    save_path = "/data/cairizhao/depthCNN/test.tfrecords"
    mat_path = '/data/cairizhao/exp/C2Re.mat'
    mat = h5.File(mat_path, 'r')
    img, depth, label = load_h5_data(mat, "TEST")
    mat.close()
    print("Making TF Record: writing {}".format(save_path))   
    np2tfrecord(img, depth,label, save_path)
    # make tf records

    
    #img = np.random.rand(1000, 256, 256, 3)
    #depth = np.random.rand(1000,32,32,1)
    #data_iterator = inputs_iterator(train=True, batch_size=64, num_epochs=10, filepath=save_path)
    #img, depth = data_iterator.get_next()
    #print('Shape of img : {}'.format(img.shape))
    #print('Shape of depth map : {}'.format(depth.shape))
    import IPython; IPython.embed()
    
   
   

    

    





    

   
