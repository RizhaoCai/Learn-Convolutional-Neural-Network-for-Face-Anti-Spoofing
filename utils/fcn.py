# This is a fully connected network for FACE SPOOFING, which predict the depth maps of of spoofing faces. 
import tensorflow as tf
import numpy as np

# variable
import numpy as np
import tensorflow as tf
import os
np.set_printoptions(formatter={'float_kind': lambda x: '%.2f' % x})


def print_vars(sess):
    print('print vars')
    for _var in tf.global_variables():
        assert _var.dtype.name == 'float32_ref', _var.name
        var = sess.run(_var)
        if _var in tf.trainable_variables():
            print('T', end='')
        else:
            print(' ', end='')
        if _var in tf.moving_average_variables():
            print('A', end='')
        else:
            print(' ', end='')
        if _var in tf.model_variables():
            print('M', end='')
        else:
            print(' ', end='')
        if _var in tf.local_variables():
            print('L', end='')
        else:
            print(' ', end='')
        print('', _var.name, var.shape, var.ravel()[0])


def weight(shape):###
    return tf.get_variable('w', shape, initializer=tf.initializers.random_normal(0.0, 0.02))###


###def w(shape):###
###    return tf.get_variable('w', shape, initializer=tf.initializers.random_normal(0.0, 0.02))###


def bias(shape):
    """
        Generate bias
    """
    return tf.get_variable('b', shape, initializer=tf.zeros_initializer())


def conv(input, c_out):
    """
        input: input from last layer
        c_out: number of the output channel 
    """
    c_in = input.get_shape()[3].value
    return tf.nn.conv2d(input, weight([3,  3, c_in, c_out]), [1, 1, 1, 1], 'SAME')


def biased_result(input):
    c_in = input.get_shape()[-1].value
    return tf.nn.bias_add(input, bias([c_in]))


def bn(input, is_training):
    ### return bias(x_flow)
    return tf.contrib.layers.batch_norm(input, is_training=is_training, updates_collections=None)


def pool(input):
    # 1，3，3，1？
    return tf.nn.max_pool(input, [1, 3, 3, 1], [1, 2, 2, 1 ], 'SAME')

def loss_layer(input, label):
    """
        L1-norm Loss
    """
    diff = tf.subtract(input, label)
    loss = tf.reduce_mean(tf.abs(diff))
    #loss = tf.norm(diff, ord=1)
    loss = tf.square(loss)
    #loss = tf.reduce_mean(norm)
    return loss
    

def create_fcn(is_training):
    with tf.variable_scope('FCN', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('X'):
            x = tf.placeholder(tf.float32, [None, 256, 256, 6])
            x_flow = x
            print('A', tf.get_default_graph().get_name_scope(), x_flow.get_shape())

        with tf.variable_scope('CONV1'):
            with tf.variable_scope('1'):
                x_flow = conv(x_flow, 64)
                x_flow = bn(x_flow, is_training)
                x_flow = tf.nn.leaky_relu(x_flow)
                print('A', tf.get_default_graph().get_name_scope(), x_flow.get_shape())
            with tf.variable_scope('2'):
                x_flow = conv(x_flow, 128)
                x_flow = bn(x_flow, is_training)
                x_flow = tf.nn.leaky_relu(x_flow)
                print('A', tf.get_default_graph().get_name_scope(), x_flow.get_shape())
            with tf.variable_scope('3'):
                x_flow = conv(x_flow, 196)
                x_flow = bn(x_flow, is_training)
                x_flow = tf.nn.leaky_relu(x_flow)
                print('A', tf.get_default_graph().get_name_scope(), x_flow.get_shape())
            with tf.variable_scope('4'):
                x_flow = conv(x_flow, 128)
                x_flow = bn(x_flow, is_training)
                x_flow = tf.nn.leaky_relu(x_flow)
                print('A', tf.get_default_graph().get_name_scope(), x_flow.get_shape())
            with tf.variable_scope('5'):
                x_flow = pool(x_flow)
                x_skip_1 = x_flow
                print('A', tf.get_default_graph().get_name_scope(), x_flow.get_shape())

        with tf.variable_scope('CONV2'):
            with tf.variable_scope('1'):
                x_flow = conv(x_flow, 128)
                x_flow = bn(x_flow, is_training)
                x_flow = tf.nn.leaky_relu(x_flow)
                print('A', tf.get_default_graph().get_name_scope(), x_flow.get_shape())
            with tf.variable_scope('2'):
                x_flow = conv(x_flow, 196)
                x_flow = bn(x_flow, is_training)
                x_flow = tf.nn.leaky_relu(x_flow)
                print('A', tf.get_default_graph().get_name_scope(), x_flow.get_shape())
            with tf.variable_scope('3'):
                x_flow = conv(x_flow, 128)
                x_flow = bn(x_flow, is_training)
                x_flow = tf.nn.leaky_relu(x_flow)
                print('A', tf.get_default_graph().get_name_scope(), x_flow.get_shape())
            with tf.variable_scope('4'):
                x_flow = pool(x_flow)
                x_skip_2 = x_flow
                print('A', tf.get_default_graph().get_name_scope(), x_flow.get_shape())

        with tf.variable_scope('CONV3'):
            with tf.variable_scope('1'):
                x_flow = conv(x_flow, 128)
                x_flow = bn(x_flow, is_training)
                x_flow = tf.nn.leaky_relu(x_flow)
                print('A', tf.get_default_graph().get_name_scope(), x_flow.get_shape())
            with tf.variable_scope('2'):
                x_flow = conv(x_flow, 196)
                x_flow = bn(x_flow, is_training)
                x_flow = tf.nn.leaky_relu(x_flow)
                print('A', tf.get_default_graph().get_name_scope(), x_flow.get_shape())
            with tf.variable_scope('3'):
                x_flow = conv(x_flow, 128)
                x_flow = bn(x_flow, is_training)
                x_flow = tf.nn.leaky_relu(x_flow)
                print('A', tf.get_default_graph().get_name_scope(), x_flow.get_shape())
            with tf.variable_scope('4'):
                x_flow = pool(x_flow)
                print('A', tf.get_default_graph().get_name_scope(), x_flow.get_shape())

        with tf.variable_scope('CAT'):
            # cat all the shortcut variables
            x_skip_1 = tf.image.resize_images(x_skip_1, [32, 32])
            x_skip_2 = tf.image.resize_images(x_skip_2, [32, 32])
            x_flow = tf.concat([x_skip_1, x_skip_2, x_flow], 3) # 3 channels?
            print('A', tf.get_default_graph().get_name_scope(), x_flow.get_shape())

        with tf.variable_scope('CONV4'):
            with tf.variable_scope('1'):
                x_flow = conv(x_flow, 128)
                x_flow = bn(x_flow, is_training)
                x_flow = tf.nn.leaky_relu(x_flow)
                print('A', tf.get_default_graph().get_name_scope(), x_flow.get_shape())
            with tf.variable_scope('2'):
                x_flow = conv(x_flow, 64)
                x_flow = bn(x_flow, is_training)
                x_flow = tf.nn.leaky_relu(x_flow)
                print('A', tf.get_default_graph().get_name_scope(), x_flow.get_shape())
            with tf.variable_scope('3'):
                x_flow = conv(x_flow, 1)
                x_flow = biased_result(x_flow)
                print('A', tf.get_default_graph().get_name_scope(), x_flow.get_shape())
        
      #  with tf.variable_scope('Loss'):
        #    d = tf.placeholder(tf.float32, [None, 32, 32, 1])
         #   l = tf.nn.sigmoid_cross_entropy_with_logits(labels=d, logits=x_flow) #loss
         #   l = tf.reduce_mean(l)  # N*32*32*1 -> 1
       #     d_hat = tf.sigmoid(x_flow)
       #     del x_flow
       
        with tf.variable_scope('L'):
            depth_gt = tf.placeholder(tf.float32, [None, 32, 32, 1]) # depth map ground truth
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=depth_gt, logits=x_flow) #loss
            loss = tf.reduce_mean(loss)  # N*32*32*1 -> 所有求平均
            #loss = loss_layer(x_flow,depth_gt)
            depth_predict = tf.sigmoid(x_flow)
            del x_flow

        if is_training:
            with tf.variable_scope('OPT'):
                train_op = tf.train.AdamOptimizer(0.00003).minimize(loss)
                #train_op = tf.train.MomentumOptimizer(0.00001, 0.9).minimize(loss) # train with momentum
                # train_op = tf.train.GradientDescentOptimizer(0.003).minimize(l)
        else:
            train_op  = tf.train.GradientDescentOptimizer(learning_rate=0.1)

    return [x, depth_gt, loss, depth_predict, train_op]


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    [x_, d_, loss_, depth_predict_, train_op_] = create_fcn(True)
    #config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
    #config.gpu_options.allow_growth = True #allocate dynamically
    #sess = tf.InteractiveSession(config = config)
    sess = tf.InteractiveSession()
    writer = tf.summary.FileWriter('logs/', sess.graph)
    sess.run(tf.global_variables_initializer())
    for i in range(100000):

        [loss, depth_predict, _] = sess.run([loss_, depth_predict_, train_op_], {x_: np.random.randn(10, 256, 256, 3), d_: np.random.randn(10, 1, 32, 32)})
        tf.contrib.layers
        print("iteration:" , i)
        print('loss:' , loss)




    






        


