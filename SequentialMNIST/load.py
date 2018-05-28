import numpy as np
import numpy.random as npr
from scipy.io import loadmat

def load_mnist(data_dir):
    fd = open("{}/train-images-idx3-ubyte".format(data_dir))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28 * 28)).astype(float)
    fd = open("{}/train-labels-idx1-ubyte".format(data_dir))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000))
    fd = open("{}/t10k-images-idx3-ubyte".format(data_dir))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28 * 28)).astype(float)
    fd = open("{}/t10k-labels-idx1-ubyte".format(data_dir))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000))
    trY = np.asarray(trY)
    teY = np.asarray(teY)
    trX = trX / 255.
    teX = teX / 255.
    vaX = trX[-10000:]
    vaY = trY[-10000:]
    trX = trX[:-10000]
    trY = trY[:-10000]
    return trX, vaX, teX, trY, vaY, teY


def load_binarized_mnist(data_path):
    # binarized_mnist_test.amat  binarized_mnist_train.amat  binarized_mnist_valid.amat
    print('loading binary MNIST, sampled version (de Larochelle)')
    train_x = np.loadtxt(data_path + '/binarized_mnist_train.amat').astype('int32')
    valid_x = np.loadtxt(data_path + '/binarized_mnist_valid.amat').astype('int32')
    test_x = np.loadtxt(data_path + '/binarized_mnist_test.amat').astype('int32')
    return train_x, valid_x, test_x
