import numpy as np

def gaussian_function(mu,sigma):
    K = 1/float(np.sqrt(2*np.pi))
    def func(x):
        return K*np.exp(-(np.dot((mu-x).T,(mu-x)))/float(2*sigma**2))
    return func

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def hinge_loss(X,Y,W):
    CW = -(Y*X.dot(W.T))
    Zeros = np.zeros(Y.shape)
    return np.maximum(Zeros,CW)
    
def hinge_loss_grad(X,Y,W):
    indicatrice = np.sign(hinge_loss(X,Y,W))
    NablaCW = indicatrice*-Y*X
    return NablaCW

def gaussian(x,mu,sigma):
    return (1/np.sqrt(2*np.pi*sigma))*np.exp(-(x-mu)**2)

class Perceptron(object):
    def __init__(self,loss=hinge_loss,loss_g=hinge_loss_grad,max_iter=100,eps=0.1):
        self.max_iter, self.eps = max_iter,eps
        self.w = None
        self.loss = loss
        self.loss_g = loss_g
        self.kernel= None
        self.initialize_histo()
        
    def initialize_histo(self):
        self.w_histo = []
        self.loss_histo = []
        
    def initialize_w(self,D):
        self.w=np.random.random((1,D))
        
    def set_kernel(self,kernel):
        self.kernel=kernel
        
    def apply_kernel(self,datax,kernel='bias'):
        N = datax.shape[0] # Number of data points
        kernel_datax = datax.reshape(N,-1)
        if kernel=='bias':
            homogeneus = np.ones((N,1))
            kernel_datax = np.hstack((datax,homogeneus))
        if kernel=='polinomial2':
            homogeneus = np.ones((N,1))
            squared = datax**2
            kernel_datax = np.hstack((squared,datax,homogeneus))
        if kernel=='rbf':
            sigma = 1.0
            kernel_datax = np.zeros((N,N))
            for i in range(N):
                kernel_datax[i,:] = np.array(map(gaussian_function(datax[i,:],sigma),datax))
        return kernel_datax
    
    def fit_minibatch_and_test(self,datax,datay,testx,testy,batches=10,kernel=None):
        datay = datay.reshape(-1,1)
        # Number of data points
        N = datax.shape[0]
        datax = self.apply_kernel(datax,kernel=kernel)
        # dimension of data points
        D = datax.shape[1] 
        self.initialize_w(D)
        self.initialize_histo()
        batch_size = N/batches
        test_loss_histo = []
        for i in range(self.max_iter):
            self.w_histo += [self.w]
            # create batch
            batch_x = datax[(i%batches)*batch_size:((i%batches)+1)*batch_size]
            batch_y = datay[(i%batches)*batch_size:((i%batches)+1)*batch_size]
            # estimate loss in train
            CW = self.loss(batch_x,batch_y,self.w)
            loss = np.sum(CW)
            self.loss_histo += [loss]
            # estimate loss in test
            TCW = self.loss(testx,testy,self.w)
            test_loss = np.sum(TCW)
            test_loss_histo += [test_loss]
            # estimate gradient
            CW_grad = self.loss_g(batch_x,batch_y,self.w)
            collective_grad = np.sum(CW_grad,axis=0) # par ligne
            self.w -= collective_grad*self.eps
        return self.loss_histo, test_loss_histo
            
    def fit_minibatch(self,datax,datay,batches=10,kernel=None):
        datay = datay.reshape(-1,1)
        N = len(datay) # Number of data points
        datax = self.apply_kernel(datax,kernel=kernel)
        D = datax.shape[1] # dimension of data points
        self.initialize_w(D)
        self.initialize_histo()
        batch_size = N/batches
        for i in range(self.max_iter):
            self.w_histo += [self.w]
            batch_x = datax[(i%batches)*batch_size:((i%batches)+1)*batch_size]
            batch_y = datay[(i%batches)*batch_size:((i%batches)+1)*batch_size]
            CW = self.loss(batch_x,batch_y,self.w)
            loss = np.sum(CW)
            self.loss_histo += [loss]
            CW_grad = self.loss_g(batch_x,batch_y,self.w)
            collective_grad = np.sum(CW_grad,axis=0) # par ligne
            self.w -= collective_grad*self.eps
            
    def fit(self,datax,datay,kernel=None):
        if kernel == None:
            kernel = self.kernel
        self.fit_minibatch(datax,datay,batches=1,kernel=kernel)
            
    def predict(self,datax,kernel=None):
        if kernel == None:
            kernel = self.kernel
        datax = self.apply_kernel(datax,kernel=kernel)
        return np.sign(datax.dot(self.w.T))
    
    def decision_function(self,datax):
        return self.predict(datax)
    
    def predict_proba(self,datax,kernel=None):
        return sigmoid(self.predict(datax))
    
    def predict_proba_distance(self,datax,kernel=None):
        return self.predict(datax)
    
    def score(self,datax,datay):
        return np.mean(self.predict(datax).T[0]==datay)