import theano
from theano import tensor as T
import numpy as np
from load import mnist
from sklearn.metrics import *
from functions import *
from load import outlier_dataset
from math import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
import csv


def para_def(input_num, layer = 5, max_h_num = 200, discount_factor = 0.25, density = 50):
    w = []
    w.append(init_weights((input_num, max_h_num)))
    m = []
    m.append(init_masks((input_num, max_h_num), density))
    n = [input_num]
    n.append(max_h_num)
    for i in range(1,layer/2):
        n.append(max(np.int(n[-1]*discount_factor), 3))
        w.append(init_weights((n[i], n[i+1])))
        m.append(init_masks((n[i], n[i+1]), density))
    #for i in range(layer/2, layer-2):
        #n.append(n[layer -1 -i-1])
        #w.append(init_weights((n[i], n[i+1])))
        #m.append(init_masks((n[i], n[i+1]), density))
    #w.append(init_weights((max_h_num, input_num)))
    #m.append(init_masks((max_h_num, input_num), density))
    #n.append(input_num)
    print n
    return (w,m)

def para_redef(w, m, layer, density):
    #density = density + np.random.randint(20, size =1)[0] - 10
    for i in range(layer/2):
        m[i].set_value(generate_masks(m[i].get_value().shape, density))
    for i in range(layer/2):
        w[i].set_value(generate_weights(w[i].get_value().shape))
    return (w,m)

def model(X, w, m, layer = 5):
    h = [X]
    #h_p = []
    h.append(T.nnet.sigmoid(T.dot(X, w[0]*m[0])))
    #h_p.append(T.nnet.sigmoid(T.dot(X, w[0])))
    for i in range(1, layer/2):
        h.append(rectify(T.dot(h[-1], w[i]*m[i])))
        #h_p.append(rectify(T.dot(h_p[-1], w[i])))
    for i in range(layer/2, layer-2):
        h.append(rectify(T.dot(h[-1], w[layer -1 -i-1].transpose()*m[layer -1 -i-1].transpose())))
        #h_p.append(rectify(T.dot(h_p[-1], w[layer -1 -i-1].transpose())))
    h.append(T.nnet.sigmoid(T.dot(h[-1], w[0].transpose()*m[0].transpose())))
    #h_p.append(T.nnet.sigmoid(T.dot(h_p[-1], w[0].transpose())))
    #return (h[-1],h_p[-1])
    return h[-1]


def model_pretrain(X, w, m, layer = 5):
    h = [X]
    h.append(T.nnet.sigmoid(T.dot(X, w[0]*m[0])))
    for i in range(1, layer/2):
        h.append(rectify(T.dot(h[-1], w[i]*m[i])))
    h_in = h[layer/2 - 1]
    for i in range(layer/2, layer-2):
        h.append(rectify(T.dot(h[-1], w[layer -1 -i-1].transpose()*m[layer -1 -i-1].transpose())))
    h.append(T.nnet.sigmoid(T.dot(h[-1], w[0].transpose()*m[0].transpose())))
    h_out = h[-(np.int(layer/2))]
    return (h_in,h_out)

#=========data-settings============
dataname = str(sys.argv[1])
datainit = int(sys.argv[2])
pp = PdfPages(dataname +'.pdf')
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
elif dataname == 'human_activity':
    trX, trY = outlier_dataset('human_activity',datainit)
    learning_rate = 0.02
elif dataname == 'pendigits':
    trX, trY = outlier_dataset('pendigits',datainit)
    learning_rate = 0.02
elif dataname == 'seismic':
    trX, trY = outlier_dataset('seismic',datainit)
    learning_rate = 0.02
elif dataname == 'thyroid':
    trX, trY = outlier_dataset('thyroid',datainit)
    learning_rate = 0.02
elif dataname == 'vowels':
    trX, trY = outlier_dataset('vowels',datainit)
    learning_rate = 0.02

#=============================

#===gerneric setting====
n_training = trX.shape[0]/3
n_iter = np.int(max(min(4*n_training, 1200),200))
density = 1.5
n_ensemble = 100
n_avg = 1
max_h_num = max(np.int(trX.shape[1]**0.75),3)
discount_factor = 0.5
layer = 7
# if trX.shape[1] <= 20:
#     layer = 5
#=======================

#====symbolic definition======
X = T.fmatrix()
lr = T.fscalar()
w, m = para_def(trX.shape[1], layer, max_h_num, discount_factor, density)
p_x  = model(X, w, m, layer)
cost = T.mean(T.sum((p_x - X)**2, axis = 1))
params = w
updates = RMSprop(cost, params, lr)
train = theano.function(inputs=[X, lr], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=p_x, allow_input_downcast=True)
#=============================

#=====pre-training============
# p_pretrain = model(X, w, m, 3)#model_pretrain(X, w[0], m[0])
# cost_pre = T.mean(T.sum((p_pretrain - X)**2, axis = 1))
# params_pre = [w[0]]
# updates_pre = RMSprop(cost_pre, params_pre, lr)
# pre_train = theano.function(inputs=[X, lr], outputs=cost_pre, updates=updates_pre, allow_input_downcast=True)

pre_train_functions = []
for i in range(0, layer/2):
    p_in, p_out = model_pretrain(X, w, m, 3 + 2*i)
    cost_pre = T.mean(T.sum((p_out - p_in)**2, axis = 1))
    params_pre = [w[i]]
    updates_pre = RMSprop(cost_pre, params_pre, lr)
    pre_train = theano.function(inputs=[X, lr], outputs=cost_pre, updates=updates_pre, allow_input_downcast=True)
    pre_train_functions.append(pre_train)
#=============================

avg_auc = 0
ensemble_err = np.zeros((n_ensemble, trX.shape[0]))
ensemble_auc = []
err_old = np.sum((predict(trX) - trX)**2, axis = 1)
err_origin = np.mean(err_old)
print err_origin
for i in range(n_avg):
    avg_err = np.zeros(err_old.shape)
    index = np.random.choice(len(trX), n_training, replace=False)
    #print len(index)
    for j in range(n_ensemble):
        l_r = learning_rate

        pretrain_index = np.random.choice(len(index), np.int(len(index)/3), replace = False)
        for fn in pre_train_functions:
            for pre_iter in range(100):
                cost_pre = fn(trX[index[pretrain_index]], l_r)

        for iter in range(n_iter):
            if iter < np.int(len(index)**0.5):
                ada_size = np.int(len(index)**0.5)
            else:
                ada_size = min(np.int(ada_size * 1.05), len(index))
            #ada_size = max(np.int(sqrt(len(index))*sqrt(sqrt(len(index)))), iter)
            train_index = np.random.choice(len(index),ada_size, replace = False)

            cost = train(trX[index[train_index]], l_r)
            # for fn in pre_train_functions:
            #     cost = fn(trX[index[train_index]], l_r)

            err = np.sum((predict(trX) - trX)**2, axis = 1)
            if np.mean(err) >= np.mean(err_old):
                l_r = l_r * 0.95
                #print 'decrese in learning Rate'
            elif np.mean(err) < np.mean(err_old) and l_r < learning_rate*1.5:
                l_r = l_r*1.002
            else:
                l_r = l_r
            err_old = err
            #print roc_auc_score(trY, err), np.mean(err), iter
        #ensemble_err.append(err)
        ensemble_err[j] = err
        #print err.shape, ensemble_err.shape
        err_norm = err / np.std(err)
        auc = roc_auc_score(trY, err_norm)
        ensemble_auc.append(auc)
        avg_err = avg_err + err_norm/n_ensemble
        print auc, np.mean(err), 'iter', iter , 'emsemble ', j
        #print roc_auc_score(trY, avg_err)
        #print m[0].get_value()
        w, m = para_redef(w, m, layer, density)
        #print m[0].get_value()
        #print np.mean(err), np.mean((predict(trX) - trX)**2)
    #avg_auc = avg_auc + roc_auc_score(trY, avg_err)/n_avg
    avg_auc = avg_auc + roc_auc_score(trY, np.median(ensemble_err, axis = 0))/n_avg
print avg_auc, 'avg_auc'

with open(dataname+'_err.csv', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in ensemble_err:
        writer.writerow(val)
    #writer.writerow([avg_auc])
    writer.writerow(trY)
with open(dataname+'.csv', "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in ensemble_auc:
        writer.writerow([val])
    writer.writerow([avg_auc])


base = np.genfromtxt(dataname+'_base.csv', delimiter=',')

plt.figure()
plt.boxplot(ensemble_auc)
plt.plot(1,base,'rs')
plt.plot(1,avg_auc,'bo')
#plt.xlim([0.0, 1.0])
#plt.ylim([0, 1])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic example')
#plt.legend(loc="lower right")
plt.savefig(pp, format='pdf')
#plt.show()
pp.close()
