import torch
import copy
def Accuracy(y,y_predict):
    leng = len(y)
    miss = 0
    for i in range(leng):
        if not y[i]==y_predict[i]:
            miss +=1
    return (leng-miss)/leng


def soft_predict(Z,temp):
    m,n = Z.shape
    Q = torch.zeros(m,n)
    Z_sum = torch.sum(torch.exp(Z/temp),dim=1)
    for i in range(n):
        Q[:,i] = torch.exp(Z[:,i]/temp)/Z_sum
    return Q

def average_weights(w):
    """
    average the weights from all local models
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg
