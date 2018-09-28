"""A script that contains many general funtions
"""
def calc_acc(y_gt,y_pred):
    """ Caculate accuracy
        y_gt and y_pred are bot 1-D vector, and they should have the same dimension

        Return the accuracy rate and the number of correct instances
    """
    num_samples = y_gt.shape[0]
     
    assert  y_gt.shape[0] == y_pred.shape[0] && num_samples>0
    num_correct = 0
    for  i in range(num_samples):
        if y_gt[i] == y_pred[i]:
            num_correct = num_correct + 1
    
    accuracy = num_correct / num_samples
    return accuracy, num_correct




