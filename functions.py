import theano
from theano import tensor as T
import numpy as np
from math import *

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(generate_weights(shape))

def generate_weights(shape):
    return floatX(np.random.randn(*shape) * 0.01)

def init_masks(shape, density):
    mask = generate_masks(shape, density)
    return theano.shared(mask)

def generate_masks(shape, density):
    mask = np.zeros(shape)
    #connections = np.int(sqrt(shape[0]*shape[1])*sqrt(sqrt(shape[0]*shape[1]))*density)
    connections = np.int(shape[0]*shape[1]*density)
    row = np.random.randint(shape[0], size = connections)
    col = np.random.randint(shape[1], size = connections)
    mask[row, col] += 1
    mask = np.double(np.greater(mask,0))
    mask = np.ones(shape)
    return mask

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
