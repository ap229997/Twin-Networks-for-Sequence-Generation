#!/bin/bash

mkdir -p mnist/data/
cd mnist/data/
wget http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat > /dev/null
wget http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat > /dev/null
wget http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat > /dev/null
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz > /dev/null
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz > /dev/null
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz > /dev/null
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz > /dev/null
gunzip *gz
