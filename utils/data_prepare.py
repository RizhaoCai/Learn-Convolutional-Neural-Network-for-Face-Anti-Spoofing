"""
    Convert numpy.arrary data to TFRecords file format
    rizhao@gmail.com
"""
import tensorflow as tf
import numpy as np


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def np2tfrecord(X, Y, label, save_path='train.tfrecords'):
    # CHECKHERE
    num_samples = X.shape[0]
    size_X =X.shape[-2]
    size_Y = Y.shape[-2]
    channel_X = X.shape[-1]
    channel_Y = Y.shape[-1]
    print('Writing', save_path)
    with tf.python_io.TFRecordWriter(save_path) as writer:
        print('Writing data to {}'.format(save_path))
        for i in range(num_samples):
            print("Transform: {} done".format(i))
            X_raw = X[i].tostring()
            Y_raw = Y[i].tostring()
            lbl = label[i]
            example = tf.train.Example( # CHECKHERE
                features=tf.train.Features(feature={
                    'X': _bytes_feature(X_raw ),
                    'Y':  _bytes_feature(Y_raw),
                    'size_X': _int64_feature(size_X),
                    'size_Y':_int64_feature( size_Y),
                    'channel_X': _int64_feature(channel_X),
                    'channel_Y': _int64_feature(channel_Y),
                    'label':_int64_feature(lbl) # int 64
                        }))

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
    # CHECKHERE
    h1, w1, c1 = 256,256,3 # Modify them to fit your case
    h2, w2, c2 = 32, 32, 1 # Modify them to fit your case
    face_shape = h1 * w1 * c1
    depth_map_shape = h2 * w2 * c2
    #print("before decode: {}".format(face_image.shape) )
    face_image.set_shape(face_shape)
    face_image=tf.reshape(face_image,[h1, w1, c1])
    depth_map.set_shape(depth_map_shape)
    depth_map=tf.reshape(depth_map,[h2, w2, c2])
    #print("after decode: {}".format(face_image.shape) )

    # It doesn't matter with int8 or int32 actually. But it helps saving memory usage with int8 in my case.
    label = tf.cast(features['label'],tf.int8)
    return face_image, depth_map, label


def inputs_iterator( dataset_path, batch_size=1, num_epochs=1,shuffle=False ):
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

    with tf.name_scope('input'):
        # the 'filename' could also be a list of filenames, which will be read in order
        dataset = tf.data.TFRecordDataset(dataset_path)
        #dataset = tf.data.TFRecordDataset([filename])
        # The map transformation taks a function and applies it to every element 
        # of the dataset
        dataset = dataset.map(decode)

        # The shuffle transformation uses a finite-sized buffer to shuffle elelemts
        # in memory. The parameter is the number of elements in the buffer. For 
        # completely uniform shuffling, set the parameter to be the same as the 
        # number of elements in the  dataset.
        if shuffle:
            dataset = dataset.shuffle(1000 + 3 * batch_size)
        # For testing or validation, 'num_epochs' is supposed to be 1
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
    return iterator



    


def build_tfrecord(save_path):
    # CHECKHERE

    # Actually, data returned here is of numpy.ndarrray
    # You should build your pipeline of loading numpy data
    img = np.zeros(shape=[1000, 256, 256, 3], dtype=np.uint8)
    depth = np.zeros(shape=[1000, 32, 32, 1], dtype=np.uint8)
    label = np.random.rand(shape=[1000,2])
    # These are dummy input
    print("Building TF Record data")
    np2tfrecord(img, depth, label, save_path)
    print('Finished!')




def how_to_embed_data_iterator_to_your_network_for_training():
    path_tfrecord = './data.tf'
    data_iterator = inputs_iterator(dataset_path=path_tfrecord, batch_size=64, num_epochs=10)

    img, depth, label = data_iterator.get_next()

    # Two ways to embed the data iterator
    # Way 1
    sess1 = tf.InteractiveSession()

    # Build the graph
    plh_x = tf.placeholder([None,256,256,3]) # PlaceHolder
    plh_y = tf.placeholder([None,2])
    k1 = tf.get_variable('w_k', [11, 11, 3, 96], initializer=tf.initializers.random_normal(0.0, 0.02))
    out = tf.nn.conv2d(plh_x, k1, [1, 1, 1, 1], 'VALID')
    out = tf.layers.flatten(out, name='fc1')
    out = tf.contrib.layers.fully_connected(out, 2, tf.nn.relu)
    loss_op = tf.losses.softmax_cross_entropy_with_glo(plh_y, out)

    # Get data
    data_x, data_y = sess.run([img, label])
    loss = sess.run([loss_op], feed_dict={plh_x:data_x, plh_y:data_y})
    sess1.close()

    # ------------------------------------------------------
    # Way2
    sess2 = tf.InteractiveSession()
    data_handle = sess2.run(data_iterator.string_handle())
    # Build the graph of the network
    plh_data_handle = tf.placeholder(tf.string, shape=[])
    # The benefit of using a placeholder for the string handle
    # is that it is flexible to change different datasets to feed your network

    embedded_iterator = tf.data.Iterator.from_string_handle(plh_data_handle, \
                                                            data_iterator.output_types, \
                                                            output_shapes=data_iterator.output_shapes
                                                            )
    embedded_img, embedded_depth, embedded_label = embedded_iterator.get_next()
    # Actually, you can put img and the label to build the graph. But it would be less flexible

    k1 = tf.get_variable('w_k', [11, 11, 3, 96], initializer=tf.initializers.random_normal(0.0, 0.02))
    out = tf.nn.conv2d(embedded_img, k1, [1, 1, 1, 1], 'VALID')
    # out = tf.nn.conv2d(img, k1, [1, 1, 1, 1], 'VALID') # This is OK, but not flexible
    out = tf.layers.flatten(out, name='fc1')
    out = tf.contrib.layers.fully_connected(out, 2, tf.nn.relu)
    loss_op = tf.losses.softmax_cross_entropy_with_glo(embedded_label, out)
    # loss_op = tf.losses.softmax_cross_entropy_with_glo(label, out) # This is OK, but not flexible

    loss = sess2.run([loss_op], feed_dict={'plh_data_handle':data_handle})
    # loss = sess.run([loss_op]) # If you don't use string
    sess.close()







if __name__ == '__main__':
    
    """
        TRAIN_X, TRAIN_D, TRAIN_LBL
        TEST_X, TEST_D, TEST_LBL
    """

    # Build tf records
    path_tfrecord = './data.tf'
    build_tfrecord(path_tfrecord)

    # Load data iterator
    data_iterator = inputs_iterator(dataset_path=path_tfrecord, batch_size=64, num_epochs=10, )
    img, depth, label = data_iterator.get_next()
    # Be noticed that they (img, depth, and label) are ops in a tensorflow graph. Neither numpy.ndarrary nor tf.tensor
    print('Shape of img : {}'.format(img.shape))
    print('Shape of depth map : {}'.format(depth.shape))

    sess = tf.InteractiveSession()
    np_img, np_depth, np_label = sess.run([img, depth, label]) # They are data of numpy.ndarrary

    import IPython; IPython.embed()
    
   
   

    

    





    

   
