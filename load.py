import numpy as np
import os

datasets_dir = 'datasets/'

def one_hot(x,n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x] = 1
	return o_h

def outlier_dataset(name = 'kddcup99', normalize = 1):
	loaded = np.genfromtxt('datasets/'+'outlier_'+name+'.csv', delimiter=',')
	n = loaded.shape[0]
	d = loaded.shape[1]
	trX = loaded[:,0:d-1]
	trY = loaded[:,d-1]

	if normalize == 1:
		trX = trX / (trX.max(axis=0)+0.00001)
	return trX, trY

def outlier_med(name = 'med', normalize = 1, hour = 2):
	loaded = np.genfromtxt('datasets/'+'outlier_'+name+'.csv', delimiter=' ')
	n = loaded.shape[0]
	d = loaded.shape[1]
	trX = loaded[:,0:d-3]
	if hour == 2:
		trY = loaded[:,d-3]
	elif hour == 3:
		trY = loaded[:,d-2]
	elif hour == 4:
		trY = loaded[:,d-1]

	if normalize == 1:
		trX = trX / (trX.max(axis=0)+0.00001)
	return trX, trY

def mnist(ntrain=60000,ntest=10000,onehot=True):
	data_dir = os.path.join('datasets/mnist/')
	fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trX = loaded[16:].reshape((60000,28*28)).astype(float)

	fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trY = loaded[8:].reshape((60000))

	fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teX = loaded[16:].reshape((10000,28*28)).astype(float)

	fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teY = loaded[8:].reshape((10000))

	trX = trX/255.
	teX = teX/255.

	trX = trX[:ntrain]
	trY = trY[:ntrain]

	teX = teX[:ntest]
	teY = teY[:ntest]

	if onehot:
		trY = one_hot(trY, 10)
		teY = one_hot(teY, 10)
	else:
		trY = np.asarray(trY)
		teY = np.asarray(teY)

	trX_raw = trX
	trY_raw = trY
	teX_raw = teX
	teY_raw = teY
	trX = np.asarray([trX_raw[i] for i in range(trY_raw.shape[0]) if trY_raw[i] == 0])
	trY = np.asarray([0 for t in trY_raw if t == 0])
	tempX = np.asarray([trX_raw[i] for i in range(trY_raw.shape[0]) if trY_raw[i] == 7])[1:101]
	tempY = np.asarray([1 for t in trY_raw if t == 7])[1:101]
	trX = np.concatenate((trX,tempX))
	trY = np.concatenate((trY,tempY))
	teX = np.asarray([teX_raw[i] for i in range(teY_raw.shape[0]) if teY_raw[i] == 0])
	teY = np.asarray([0 for t in teY_raw if t == 0])
	trX = np.concatenate((trX,teX))
	trY = np.concatenate((trY,teY))

	return trX,trY
