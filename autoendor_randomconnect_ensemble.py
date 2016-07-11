import theano
from theano import tensor as T
import numpy as np
from load import mnist
from sklearn.metrics import *
from functions import *
from load import outlier_dataset
from math import floor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys

def para_def(input_num, layer = 5, max_h_num = 200, discount_factor = 0.25, density = 50):
    w = []
    w.append(init_weights((input_num, max_h_num)))
    m = []
    m.append(init_masks((input_num, max_h_num), density))
    n = [input_num]
    n.append(max_h_num)
    for i in range(1,layer/2):
        n.append(np.int(n[-1]*discount_factor))
        w.append(init_weights((n[i], n[i+1])))
        m.append(init_masks((n[i], n[i+1]), density))
    for i in range(layer/2, layer-2):
        n.append(n[layer -1 -i-1])
        w.append(init_weights((n[i], n[i+1])))
        m.append(init_masks((n[i], n[i+1]), density))
    w.append(init_weights((max_h_num, input_num)))
    m.append(init_masks((max_h_num, input_num), density))
    n.append(input_num)
    print n
    return (w,m)

def para_redef(w, m, density):
    #density = density + np.random.randint(20, size =1)[0] - 10
    for m_i in m:
        m_i.set_value(generate_masks(m_i.get_value().shape, density))
    for w_i in w:
        w_i.set_value(generate_weights(w_i.get_value().shape))
    return (w,m)

def model(X, w, m, layer = 5):
    h = []
    h_p = []
    h.append(T.nnet.sigmoid(T.dot(X, w[0]*m[0])))
    h_p.append(T.nnet.sigmoid(T.dot(X, w[0])))
    for i in range(1, layer/2):
        h.append(rectify(T.dot(h[-1], w[i]*m[i])))
        h_p.append(rectify(T.dot(h_p[-1], w[i])))
    for i in range(layer/2, layer-2):
        h.append(rectify(T.dot(h[-1], w[i]*m[i])))
        h_p.append(rectify(T.dot(h_p[-1], w[i])))
    h.append(T.nnet.sigmoid(T.dot(h[-1], w[layer-2]*m[layer-2])))
    h_p.append(T.nnet.sigmoid(T.dot(h_p[-1], w[layer-2])))
    return (h[-1],h_p[-1])

#=========data-settings============
dataname = str(sys.argv[1])
datainit = int(sys.argv[2])
pp = PdfPages(dataname +'.pdf')
#=============================

#=========dataset=============
if dataname == 'mnist':
    trX, trY = mnist(onehot=False)
    layer = 5
    max_h_num = 200
    discount_factor = 0.25
    density = 40
    learning_rate = 0.02
    n_iter = 100
    n_ensemble = 100
    n_avg = 1
elif dataname == 'cardio':
    trX, trY = outlier_dataset('cardio',datainit)
elif dataname == 'lympho':
    trX, trY = outlier_dataset('lympho',datainit) #small one
elif dataname == 'ecoli':
    trX, trY = outlier_dataset('ecoli',datainit)
elif dataname == 'musk':
    trX, trY = outlier_dataset('musk',1)
    layer = 5
    max_h_num = 200
    discount_factor = 0.25
    density = 40
    learning_rate = 0.02
    n_iter = 100
    n_ensemble = 100
    n_avg = 1
elif dataname == 'optdigits':
    trX, trY = outlier_dataset('optdigits',1)
    layer = 5
    max_h_num = 200
    discount_factor = 0.25
    density = 30
    learning_rate = 0.02
    n_iter = 1000
    n_ensemble = 100
    n_avg = 1
elif dataname == 'waveform':
    trX, trY = outlier_dataset('waveform',1)
    layer = 5
    max_h_num = 50
    discount_factor = 0.25
    density = 10
    learning_rate = 0.02
    n_iter = 1000
    n_ensemble = 100
    n_avg = 1
elif dataname == 'yeast':
    trX, trY = outlier_dataset('yeast',0) #small one
    layer = 5
    max_h_num = 50
    discount_factor = 0.25
    density = 5
    learning_rate = 0.01
    n_iter = 300
    n_ensemble = 100
    n_avg = 1
elif dataname == 'kddcup99':
    trX, trY = outlier_dataset('kddcup99',1)
    layer = 5
    max_h_num = 50
    discount_factor = 0.25
    density = 10
    learning_rate = 0.02
    n_iter = 300
    n_ensemble = 100
    n_avg = 1
#=============================

#====symbolic definition======
X = T.fmatrix()
lr = T.fscalar()
w, m = para_def(trX.shape[1], layer, max_h_num, discount_factor, density)
p_x, p_x_predict = model(X, w, m, layer)
cost = T.mean(T.sum((p_x - X)**2, axis = 1))
params = w
updates = RMSprop(cost, params, lr)
train = theano.function(inputs=[X, lr], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=p_x_predict, allow_input_downcast=True)
#=============================

avg_auc = 0
ensemble_auc = []
err_old = np.sum((predict(trX) - trX)**2, axis = 1)
for i in range(n_avg):
    avg_err = np.zeros(err_old.shape)
    index = np.random.randint(0, len(trX),[trX.shape[0]/10,])
    for j in range(n_ensemble):
        l_r = learning_rate
        for iter in range(n_iter):
            cost = train(trX[index], l_r)
            err = np.sum((predict(trX) - trX)**2, axis = 1)
            if np.mean(err) >= np.mean(err_old):
                l_r = l_r * 0.95
                #print 'decrese in learning Rate'
            elif np.mean(err) < np.mean(err_old) and l_r < learning_rate*1.5:
                l_r = l_r*1.002
            else:
                l_r = l_r
            err_old = err
            print roc_auc_score(trY, err), np.mean(err), iter
        avg_err = avg_err + err/n_ensemble
        auc = roc_auc_score(trY, err)
        ensemble_auc.append(auc)
        print auc, np.mean(err), 'emsemble ', j
        #print m[0].get_value()
        w, m = para_redef(w, m, density)
        #print m[0].get_value()
        #print np.mean(err), np.mean((predict(trX) - trX)**2)
    avg_auc = avg_auc + roc_auc_score(trY, avg_err)/n_avg
print avg_auc, 'avg_auc'


plt.figure()
plt.boxplot(ensemble_auc)
plt.plot(1,avg_auc,'bo')
#plt.xlim([0.0, 1.0])
#plt.ylim([0, 1])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic example')
#plt.legend(loc="lower right")
plt.show()
plt.savefig(pp, format='pdf')
pp.close()
