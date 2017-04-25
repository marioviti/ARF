import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def show_usps(data,size=6):
    """
    plot figure of digit
    """
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")
    plt.show()

def char(character,datax,datay):
    """
    extract class
    """
    return datax[datay==character],datay[datay==character]


def plot_ROC(roc_curve_values, axe=plt,title='Receiver operating characteristic example'):
    fpr,tpr,roc_auc = roc_curve_values
    #axe.figure()
    axe.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    axe.plot([0, 1], [0, 1], color='navy', linestyle='--')
    axe.xlim([0.0, 1.0])
    axe.ylim([0.0, 1.05])
    axe.xlabel('False Positive Rate')
    axe.ylabel('True Positive Rate')
    axe.title(title)
    axe.legend(loc="lower right")

def compute_ROC(X,Y,clf):
    """
    Compute ROC of a trained classifier
    """
    y_score = clf.decision_function(X)
    y_score = label_binarize(y_score,classes=[0,1],  neg_label=-1, pos_label=1 )
    fpr,tpr,_ = roc_curve(Y, y_score)
    roc_auc = auc(fpr,tpr)
    return fpr,tpr,roc_auc
    

def usps_1vsMulti_class_train_and_test(trainx,trainy,testx,testy,clf,classes = 10):
    """
    Multiclass classification using 1 vs multi
    """
    print 'trainig, using 1 vs multi'
    train_scores = np.zeros(classes)
    test_scores = np.zeros(classes)
    roc_curves = {}
    w_histo = []
    for i in range(classes):
        # extract positives aka 1
        train_datax,train_datay = char(i,trainx,trainy)
        test_datax,test_datay = char(i,testx,testy)
        # extract positives test labels
        test_datay = np.ones(test_datay.shape)
        # extract positives train labels
        train_datay = np.ones(train_datay.shape) # create ones
        
        for j in range(classes):            
            if not i==j:
                # extract train negatives examples and labels aka -1 aka "multi"
                train_ch1x,train_ch1y = char(j,trainx,trainy) 
                # stacking vertically on top negatives examples to the train and test set
                train_datax = np.vstack((train_datax,train_ch1x))
                # stacking horizontally to the left  negatives labels to the train and test set
                train_datay = np.hstack((train_datay,np.zeros(train_ch1y.shape)-1)) # add -1
                
                # extract train negatives examples and labels aka -1 aka "multi"
                test_ch1x,test_ch1y = char(j,testx,testy) 
                test_datax = np.vstack((test_datax,test_ch1x))
                test_datay = np.hstack((test_datay,np.zeros(test_ch1y.shape)-1)) # add -1
        
        # training the classifier
        clf.fit(train_datax,train_datay)
        # getting the score on train data
        train_scores[i] = clf.score(train_datax,train_datay)
        # getting the score on test data
        test_scores[i] = clf.score(test_datax,test_datay)
        # roc curve data
        train_datay = label_binarize(train_datay, classes=[-1, 1])
        test_datay = label_binarize(test_datay, classes=[-1, 1])
        roc_curves[i] = compute_ROC(test_datax,test_datay,clf)
        # save the weight
        w_histo += [clf.w.copy()]
        
    return train_scores, test_scores, roc_curves, w_histo

def usps_1vs1_class_trant_and_test(trainx,trainy,testx,testy,clf,classes = 10):
    """
    Multiclass classification using 1 vs 1
    """
    train_scores = np.zeros((classes,classes))
    test_scores = np.zeros((classes,classes))
    for i in range(classes):
        for j in range(classes):
            datax = None
            datay = None
            if not i==j:
                ch0x,ch0y = char(i,trainx,trainy)
                ch1x,ch1y = char(j,trainx,trainy) 
                train_datax = np.vstack((ch0x,ch1x))
                train_datay = np.hstack((np.zeros(ch1y.shape)-1,np.zeros(ch0y.shape)+1))
                
                testch0x,testch0y = char(i,testx,testy)
                testch1x,testch1y = char(j,testx,testy)
                test_datax = np.vstack((testch0x,testch1x))
                test_datay = np.hstack((np.zeros(testch1y.shape)-1,np.zeros(testch0y.shape)+1))

                clf.fit(train_datax,train_datay)
                train_scores[i,j] = clf.score(train_datax,train_datay)
                test_scores[i,j] = clf.score(test_datax,test_datay)
                y_scores = clf.decision_function(testch0x)
    return train_scores, test_scores