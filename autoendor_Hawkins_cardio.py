import theano
from theano import tensor as T
import numpy as np
from load import mnist
from load import outlier_dataset
from sklearn.metrics import *
#import matplotlib.pyplot as plt

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)

def sgd(cost, params, lr):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates

def stairactivation(theta, k=3, N=4, a=100):
    tmp = 0
    for j in range(1, N):
        tmp = tmp + T.tanh(a*(theta - j/N))
    return 0.5 + tmp/2/k

def model(X, w_h1, w_h2, w_h3, w_h3b, w_h2b, w_h1b):
    h1 = T.tanh(T.dot(X, w_h1))
    h2 = T.tanh(T.dot(h1, w_h2))
    h3 = stairactivation(T.dot(h2, w_h3), 3, 4, 10)
    h4 = T.tanh(T.dot(h3, w_h3b))
    h5 = T.tanh(T.dot(h4, w_h2b))
    px = T.nnet.sigmoid(T.dot(h5, w_h1b))
    return px

trX, trY = outlier_dataset('cardio',0)

X = T.fmatrix()
lr = T.fscalar()


s_1 = (21, 200)
s_2 = (200, 50)
s_3 = (50,10)
s_3b = tuple(reversed(s_3))
s_2b = tuple(reversed(s_2))
s_1b = tuple(reversed(s_1))

w_h1 = init_weights(s_1)
w_h2 = init_weights(s_2)
w_h3 = init_weights(s_3)
w_h3b = init_weights(s_3b)
w_h2b = init_weights(s_2b)
w_h1b = init_weights(s_1b)

p_x = model(X, w_h1, w_h2, w_h3, w_h3b, w_h2b, w_h1b)

cost = T.mean(T.sum((p_x - X)**2, axis = 1))
params = [w_h1, w_h2, w_h3, w_h3b, w_h2b, w_h1b]
updates = sgd(cost, params, lr)

train = theano.function(inputs=[X, lr], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=p_x, allow_input_downcast=True)

err_old = np.mean((predict(trX) - trX)**2)

l_r = 0.1
for i in range(1):
    for iter in range(500):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
            cost = train(trX[start:end], l_r)
        err = np.mean(np.sum((predict(trX) - trX)**2, axis = 1))
        print roc_auc_score(trY, np.sum((predict(trX) - trX)**2, axis = 1)), err, iter
        if err >= err_old:
            l_r = l_r * 0.98
            #print 'dfdf'
        elif err < err_old and l_r < 0.1:
            l_r = l_r*1.002
        else:
            l_r = l_r
        err_old = err





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
