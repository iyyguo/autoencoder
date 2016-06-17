import theano
from theano import tensor as T
import numpy as np
from load import mnist
from sklearn.metrics import *
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
#from matplotlib.pyplot import *
import matplotlib.pyplot as plt
#from foxhound.utils.vis import grayscale_grid_vis, unit_scale
#from scipy.misc import imsave

srng = RandomStreams()

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

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

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

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

def model(X, w_h1, w_h2, w_h3, w_o, p_drop_input, p_drop_hidden):
    h1 = T.nnet.sigmoid(dropout(T.dot(X, w_h1), p_drop_input))
    h2 = rectify(dropout(T.dot(h1, w_h2), p_drop_hidden))
    h3 = rectify(dropout(T.dot(h2, w_h3), p_drop_hidden))
    px = T.nnet.sigmoid(dropout(T.dot(h3, w_o), p_drop_hidden))
    ##h3 = stairactivation(T.dot(h2, w_h3), 3, 4, 10)
    return px

trX, trY = mnist(onehot=False)

X = T.fmatrix()
lr = T.fscalar()

w_h1 = init_weights((784, 200))
w_h2 = init_weights((200, 30))
w_h3 = init_weights((30, 200))
w_o = init_weights((200, 784))

p_x = model(X, w_h1, w_h2, w_h3, w_o, 0.2, 0.5)
p_x_predict = model(X, w_h1, w_h2, w_h3, w_o, 0, 0)

cost = T.mean(T.sum((p_x - X)**2, axis = 1))
params = [w_h1, w_h2, w_h3, w_o]
updates = RMSprop(cost, params, lr)

train = theano.function(inputs=[X, lr], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=p_x_predict, allow_input_downcast=True)

err_old = np.mean((predict(trX) - trX)**2, axis = 1)

l_r = 0.001
n_ensemble = 100
for N in range(n_ensemble):
    avg_err = np.zeros(err_old.shape)
    for iter in range(5):
        index = np.random.randint(0, len(trX),[1280,])
        cost = train(trX[index], l_r)
        err = np.mean((predict(trX) - trX)**2, axis = 1)
        if np.mean(err) >= np.mean(err_old):
            l_r = l_r * 0.95
            #print 'decrese in learning Rate'
        elif np.mean(err) < np.mean(err_old) and l_r < 0.1:
            l_r = l_r*1.002
        else:
            l_r = l_r
        err_old = err
        avg_err = avg_err + err/n_ensemble
    print roc_auc_score(trY, avg_err), np.mean(err), iter
    w_h1 = init_weights((784, 200))
    w_h2 = init_weights((200, 30))
    w_h3 = init_weights((30, 200))
    w_o = init_weights((200, 784))





fpr, tpr, _ = roc_curve(trY, np.mean((predict(trX) - trX)**2, axis = 1))
plt.figure()
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()
