"""
In the training of CNN, the learning rate is 0.001; decay rate is 0.001; and the momentum during training is 0.9.
Before fed into the CNN, the data are first centralized by the mean of training data. 
These parameters are constant in our experiments. 
"""

import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import alexnet
import numpy as np
import utils.my_utils as utils
import utils.CNN as CNN

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

class AlexNet():
    def __init__(self, num_class=2,image_shape=[128,128,3]):
        s_ = [None]
        s_.extend(image_shape)
        self.x_shape = s_
        print("Input shape:",self.x_shape )
        self.x_input = tf.placeholder(dtype=tf.float32, shape=self.x_shape)
        self.y_gt = tf.placeholder(dtype=tf.int8, shape=[None,2], name='y_ground_truth')
        self.num_class = num_class
        self.init = False    
        self.model = self.alex_net(True)
        self.__learning_rate = 0.001
        #self.__self
        self.__batch_size = 25
        # build Alex-net
        self.__log_path = 'log'

    def set_hyper_parameters(self,batch_size, learning_rate):
         self.__learning_rate = learning_rate
    
    def alex_net(self, is_traning=True):
        print("Building Alex-net")
        with tf.variable_scope("AlexNet", reuse=tf.AUTO_REUSE):
            with tf.variable_scope('input'):
                x = self.x_input
                
                print('A', tf.get_default_graph().get_name_scope(), x.get_shape())

            with tf.variable_scope("Conv-1"):
                k1 = tf.get_variable('w_k',[11,11,3,96] ,initializer=tf.initializers.random_normal(0.0, 0.02))
                x = tf.nn.conv2d(x, k1, [1,1,1,1],'VALID')
                x  = tf.nn.leaky_relu(x)
                x = tf.nn.lrn(x) # todo: add more detailed parameters.  alpha = 0.0001, beta = 0.75, knorm = 2
                x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1 ], 'SAME')
                print('A', tf.get_default_graph().get_name_scope(), x.get_shape())

            with tf.variable_scope("Conv-2"):
                in_channels = x.get_shape()[-1].value
                out_channels = 256
                k2  = tf.get_variable('w_k',[5,5, in_channels, out_channels] ,initializer=tf.initializers.random_normal(0.0, 0.02))
                x = tf.nn.conv2d(x, k2, [1,1,1,1],'SAME')
                x  = tf.nn.leaky_relu(x)
                x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1 ], 'SAME')
                print('A', tf.get_default_graph().get_name_scope(), x.get_shape())

            with tf.variable_scope("Conv-3"):
                in_channels = x.get_shape()[-1].value
                out_channels = 384
                k3  = tf.get_variable('w_k',[3,3, in_channels, out_channels] ,initializer=tf.initializers.random_normal(0.0, 0.02))
                x = tf.nn.conv2d(x, k3, [1,1,1,1],'SAME')
                print('A', tf.get_default_graph().get_name_scope(), x.get_shape())

            with tf.variable_scope("Conv-4"):
                in_channels = x.get_shape()[-1].value
                out_channels = 384
                k4  = tf.get_variable('w_k',[3,3, in_channels, out_channels] ,initializer=tf.initializers.random_normal(0.0, 0.02))
                x = tf.nn.conv2d(x, k4, [1,1,1,1],'SAME')
                print('A', tf.get_default_graph().get_name_scope(), x.get_shape())

            with tf.variable_scope("Conv-5"):
                in_channels = x.get_shape()[-1].value
                out_channels = 256
                k5  = tf.get_variable('w_k',[3,3, in_channels, out_channels], initializer=tf.initializers.random_normal(0.0, 0.02))
                x = tf.nn.conv2d(x, k5, [1,1,1,1],'SAME')
                x  = tf.nn.leaky_relu(x)
                x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1 ], 'SAME')
                print('A', tf.get_default_graph().get_name_scope(), x.get_shape())
            
            with  tf.variable_scope("FC1"):
                x = tf.layers.flatten(x,name='fc1')
                num_output = 3
                x = tf.contrib.layers.fully_connected(x, num_output, tf.nn.relu)
                keep_prob = 0.5
                x = tf.nn.dropout(x, keep_prob)
                print('A', tf.get_default_graph().get_name_scope(), x.get_shape())
            with tf.variable_scope("FC2"):
                x = tf.contrib.layers.fully_connected(x, num_output, tf.nn.relu, )
                keep_prob = 0.5
                x = tf.nn.dropout(x, keep_prob)
                print('A', tf.get_default_graph().get_name_scope(), x.get_shape())
            with tf.variable_scope("FC3"):
                num_class = 2
                x = tf.contrib.layers.fully_connected(x, num_class, tf.nn.relu )     
                # this can be used for feature extraction
                print('A', tf.get_default_graph().get_name_scope(), x.get_shape())
                softmax = tf.nn.softmax(x)
                
            with tf.variable_scope("Loss"):
                loss = tf.losses.softmax_cross_entropy(self.y_gt, softmax)
            with tf.variable_scope("Optimization"):
                train_op = tf.train.AdamOptimizer().minimize(loss)

            return {"softmax_op": softmax, "loss_op": loss, "train_op": train_op}
    def train(self,x_train, y_train, sess ):
        
        train_op = self.model["train_op"] 
        loss_op = self.model["loss_op"]
        softmax_op = self.model["softmax_op"]
        softmax, loss, _ = sess.run([softmax_op, loss_op, train_op], feed_dict={self.x_input:x_train,self.y_gt:y_train})
        return softmax, loss

    def test(self, x_test, y_test, sess):
        """ Test/evaluate the model
            Make sure that the valuables is not initialized by the initilizer
        """
        softmax = self.model["softmax"]
        pred_label = tf.argmax(softmax, axis=1)
        pred_result = sess.run(pred_label,feed_dict={self.x_input:x_test,self.y_gt:y_test})

        acc, num_correct = utils.calc_acc(y_test,pred_result )
        print("Testing samples: {}. Correct Samples: {}. \
                 Accuracy rate: {}".format(y_test.shape[0],num_correct,acc))


    def predict(self, x_2pred,sess):
        # TODO: Implement the code of prediction
        pass
    def restore(self, ckpt_filepath, sess):
        if ckpt_filepath == None:
            print("Initilize all variables with tf.initializer")
            sess.run(tf.initialize_all_variables())
            return False
        try:
            saver = tf.train.Saver()
            saver.restore(sess2, ckpt_filepath)
        except:
            print("Errors occur when loading check point file")
            return False
        else:
            print("Initilize all variables by loading checkpoint file")
            return True
    def save(self, save_path,sess):
        saver = tf.train.Saver()
        #if os.path.exists(logs):
        #    os.makedirs('save_path') # todo : code more robustly
        try:
            saver.save(sess, save_path) 
        except:
            print("Failed save model to path: {}".format(save_path))
            return False
        print("Sucessfully save model to path: {}".format(save_path))
        return True
    


    

def main():
    
    X = np.random.rand(10, 32, 32, 3)
    y = np.array([[1,0], [0,1],[1,0], [0,1],[1,0], [0,1],[1,0], [0,1],[1,0], [0,1]])
    batch_size = X.shape[0]
    alex = AlexNet(2,X.shape[1:])
    sess = tf.InteractiveSession()
    alex.restore(None, sess)
    for i_ in range(10):
        softmax, loss = alex.train(X,y,sess)
        print("Iteration: {}; Softmax result: {}; Loss： {}".format(i_, softmax, loss))

if __name__ == '__main__':
    main()
