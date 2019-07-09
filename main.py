import sys
sys.append('.\')
import tensorflow as tf
import alexnet as net
import utils.data_prepare as data
def train():
    sess = tf.InteractiveSession()
    model = net.AlexNet()
    tf_record_path = './data.tf' # This should be prepared in advance
    data_iterator = data(tf_record_path, num_epochs=2)
    X_train_it, _, y_train_it = data_iterator.get_next()
    X_train, _y_train = sess.run([X_train_it, y_train_it])
    model.train(X_train, y_train, sess)
    
def test():
    pass # Not Implemented
if __name__ == '__main__':
    train()
