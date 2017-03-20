#from arftools import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")


class Perceptron(object):
    def __init__(self,loss,loss_g,max_iter=100,eps=0.01):
        self.max_iter, self.eps = max_iter,eps
        self.w = None
        self.w_histo,self.loss_histo = None,None
        self.loss = loss
        self.loss_g = loss_g

    def fit(self,datax,datay):
        datay = datay.reshape(-1,1)
        N = len(datay)
        datax = datax.reshape(N,-1)
        D = datax.shape[1]
        self.w = np.random.random((1,D))
        self.w_histo = []
        self.loss_histo = []
        for i in range(self.max_iter):
            pass
    def predict(self,datax):
        pass
    def score(self,datax,datay):
        return np.mean(self.predict(datax)==datay)
