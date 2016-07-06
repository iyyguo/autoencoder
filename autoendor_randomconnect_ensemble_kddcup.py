import theano
from theano import tensor as T
import numpy as np
from load import mnist
from load import outlier_dataset
from sklearn.metrics import *

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    weight = floatX(np.random.randn(*shape) * 0.01)
    #print weight
    return theano.shared(weight)

def init_masks(shape, sparse = 10):
    mask = np.zeros(shape)
    row = np.random.randint(shape[0], size = shape[0]*shape[1]/sparse)
    col = np.random.randint(shape[1], size = shape[0]*shape[1]/sparse)
    mask[row, col] += 1
    #print mask
    return theano.shared(mask)

def reinit_masks(shape, sparse = 10):
    mask = np.zeros(shape)
    row = np.random.randint(shape[0], size = shape[0]*shape[1]/sparse)
    col = np.random.randint(shape[1], size = shape[0]*shape[1]/sparse)
    mask[row, col] += 1
    return mask

def rectify(X):
    return T.maximum(X, 0.)

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def sgd(cost, params, lr):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates

def model(X, w_h1, w_h2, w_h3, w_o, m_h1, m_h2, m_h3, m_o):
    h1 = T.nnet.sigmoid(T.dot(X, m_h1*w_h1))
    h2 = rectify(T.dot(h1, m_h2*w_h2))
    h3 = rectify(T.dot(h2, m_h3*w_h3))
    px = T.nnet.sigmoid(T.dot(h3, m_o*w_o))
    return px

trX, trY = outlier_dataset('kddcup99')

X = T.fmatrix()
lr = T.fscalar()

s_1 = (41, 200)
s_2 = (200, 20)
s_3 = tuple(reversed(s_2))
s_4 = tuple(reversed(s_1))

w_h1 = init_weights(s_1)
w_h2 = init_weights(s_2)
w_h3 = init_weights(s_3)
w_o = init_weights(s_4)
m_h1 = init_masks(s_1)
m_h2 = init_masks(s_2)
m_h3 = init_masks(s_3)
m_o = init_masks(s_4)

p_x = model(X, w_h1, w_h2, w_h3, w_o, m_h1, m_h2, m_h3, m_o)
p_x_predict = model(X, w_h1, w_h2, w_h3, w_o, m_h1, m_h2, m_h3, m_o)

cost = T.mean(T.sum((p_x - X)**2, axis = 1))
params = [w_h1, w_h2, w_h3, w_o]
updates = RMSprop(cost, params, lr)

train = theano.function(inputs=[X, lr], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=p_x_predict, allow_input_downcast=True)

err_old = np.mean((predict(trX) - trX)**2, axis = 1)


n_ensemble = 100
n_avg = 1
avg_auc = 0
for i in range(n_avg):
    avg_err = np.zeros(err_old.shape)
    index = np.random.randint(0, len(trX),[10000,])
    for m in range(n_ensemble):
        l_r = 0.1
        for iter in range(100):
            cost = train(trX[index], l_r)
            err = np.mean((predict(trX) - trX)**2, axis = 1)
            err_train = np.mean((predict(trX[index]) - trX[index])**2, axis = 1)
            if np.mean(err) >= np.mean(err_old):
                l_r = l_r * 0.95
                #print 'decrese in learning Rate'
            elif np.mean(err) < np.mean(err_old) and l_r < 0.1:
                l_r = l_r*1.002
            else:
                l_r = l_r
            err_old = err
            print roc_auc_score(trY, err), roc_auc_score(trY[index], err_train), np.mean(err), iter
            #print w_h1.get_value()
        avg_err = avg_err + err/n_ensemble
        auc = roc_auc_score(trY, err)
        print auc, np.mean(err), 'emsemble ', m
        #print w_h1.get_value()
        w_h1.set_value(floatX(np.random.randn(*s_1) * 0.01))
        #print w_h1.get_value()
        w_h2.set_value(floatX(np.random.randn(*s_2) * 0.01))
        w_h3.set_value(floatX(np.random.randn(*s_3) * 0.01))
        w_o.set_value(floatX(np.random.randn(*s_4) * 0.01))
        m_h1.set_value(reinit_masks(s_1))
        m_h2.set_value(reinit_masks(s_2))
        m_h3.set_value(reinit_masks(s_3))
        m_o.set_value(reinit_masks(s_4))
        #print np.mean(err), np.mean((predict(trX) - trX)**2)
    avg_auc = avg_auc + roc_auc_score(trY, avg_err)/n_avg
print avg_auc, 'avg_auc'
