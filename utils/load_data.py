# 
import h5py as h5

def load_data(filepath,set_type,stride=1):
    mat = h5.File(filepath, 'r')
    if set_type=='TRAIN':
        X = mat['TRAIN_X'][::stride]
        depth_map = mat['TRAIN_D'][::stride]
        label = mat['TRAIN_LBL'][::stride]
    if set_type == 'TEST':
        X = mat['TEST_X'][::stride]
        depth_map = mat['TEST_D'][::stride]
        label = mat['TEST_LBL'][::stride]
    mat.close()
    return X, depth_map, label
    
if __name__ == "__main__":
    filepath = '/data/cairizhao/exp/C2RTT.mat'
    X, D = load_data(filepath, 'TRAIN')
    print(X.shape)
    print(type(X))
