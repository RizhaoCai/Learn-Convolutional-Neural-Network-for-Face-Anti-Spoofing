import sys
sys.append('.\')
import tensorflow as tf
import alexnet as net
import utils.data_utils as data
def train():
    dataset_path = ""
    sess = tf.InteractiveSession()
    model = net.AlexNet()
    stride = 100
    X_train, y_train = data.load_data(dataset_path,"TRAIN", stride)
    model.train(X_train, y_train, sess)
    
def test():
    pass
if __name__ == '__main__':
    train()