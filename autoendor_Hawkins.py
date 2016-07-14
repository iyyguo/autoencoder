import theano
from theano import tensor as T
import numpy as np
from load import mnist
from sklearn.metrics import *
from functions import *
from load import outlier_dataset
from math import floor
#import matplotlib.pyplot as plt
import sys
import csv

def para_def(input_num, layer = 5, max_h_num = 200, discount_factor = 0.25):
    w = []
    w.append(init_weights((input_num, max_h_num)))
    n = [input_num]
    n.append(max_h_num)
    for i in range(1,layer/2):
        n.append(np.int(n[-1]*discount_factor))
        w.append(init_weights((n[i], n[i+1])))
    for i in range(layer/2, layer-2):
        n.append(n[layer -1 -i-1])
        w.append(init_weights((n[i], n[i+1])))
    w.append(init_weights((max_h_num, input_num)))
    return w

def model(X, w, layer = 5):
    h = []
    h.append(T.tanh(T.dot(X, w[0])))
    for i in range(1, layer/2):
        if i == layer/2 -1:
            h.append(stairactivation(T.dot(h[-1], w[i]), 3, 4, 10))
        else:
            h.append(T.tanh(T.dot(h[-1], w[i])))
    for i in range(layer/2, layer-2):
        h.append(T.tanh(T.dot(h[-1], w[i])))
    h.append(T.nnet.sigmoid(T.dot(h[-1], w[layer-2])))
    return h[-1]


#=========data-settings============
dataname = str(sys.argv[1])
datainit = int(sys.argv[2])
#=============================

#=========dataset=============
if dataname == 'mnist':
    trX, trY = mnist(onehot=False)
    learning_rate = 0.02
elif dataname == 'cardio':
    trX, trY = outlier_dataset('cardio',datainit)
    learning_rate = 0.02
elif dataname == 'lympho':
    trX, trY = outlier_dataset('lympho',datainit) #small one
    learning_rate = 0.02
elif dataname == 'ecoli':
    trX, trY = outlier_dataset('ecoli',datainit)
    learning_rate = 0.02
elif dataname == 'musk':
    trX, trY = outlier_dataset('musk',datainit)
    learning_rate = 0.02
elif dataname == 'optdigits':
    trX, trY = outlier_dataset('optdigits',datainit)
    learning_rate = 0.02
elif dataname == 'waveform':
    trX, trY = outlier_dataset('waveform',datainit)
    learning_rate = 0.02
elif dataname == 'yeast':
    trX, trY = outlier_dataset('yeast',datainit) #small one
    learning_rate = 0.01
elif dataname == 'kddcup99':
    trX, trY = outlier_dataset('kddcup99',datainit)
    learning_rate = 0.02
elif dataname == 'gisette':
    trX, trY = outlier_dataset('gisette',datainit)
    learning_rate = 0.05
#=============================

#===gerneric setting====
n_training = trX.shape[0]/10
n_iter = 300
density = 1
n_ensemble = 100
n_avg = 1
max_h_num = max(np.int(trX.shape[1]**0.75),3)
discount_factor = 0.25
layer = 5
l_r = learning_rate
#=======================





X = T.fmatrix()
lr = T.fscalar()
w = para_def(trX.shape[1], layer, max_h_num, discount_factor)
p_x = model(X, w, layer)
cost = T.mean(T.sum((p_x - X)**2, axis = 1))
params = w
updates = sgd(cost, params, lr)
train = theano.function(inputs=[X, lr], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=p_x, allow_input_downcast=True)

err_old = np.mean(np.sum((predict(trX) - trX)**2, axis = 1))
for i in range(n_avg):
    for iter in range(n_iter):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
            cost = train(trX[start:end], l_r)
        err = np.mean(np.sum((predict(trX) - trX)**2, axis = 1))
        auc = roc_auc_score(trY, np.sum((predict(trX) - trX)**2, axis = 1))
        print auc , err, iter
        if err >= err_old:
            l_r = l_r * 0.98
            print 'dfdf'
        elif err < err_old and l_r < 0.1:
            l_r = l_r*1.002
        else:
            l_r = l_r
        err_old = err

with open(dataname+'_base.csv', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow([auc])



# fpr, tpr, _ = roc_curve(trY, np.mean((predict(trX) - trX)**2, axis = 1))
# plt.figure()
# plt.plot(fpr, tpr, label='ROC curve')
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()
