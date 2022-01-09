# SimCLR
A jupyter-notebook implementation of SimCLR on CIFAR-10 for playing with 
semi-supervised learning using a small percentage of labeled CIFAR-10 
training data and transfer learning by applying to the CIFAR-100 dataset

Some modest benchmarks so far, following 50 epochs of SimCLR training:
63% validation accuracy on CIFAR-10 when only using 1% labeled data
(with frozen feature net)

37% top-1 classification on CIFAR-100 (with frozen feature net

Implementation of SimCLR adapted from implementations at 
https://github.com/Spijkervet/SimCLR
https://github.com/sadimanna/simclr_pytorch

Py files:

resnet50.py
--Generates a modified resnet-50 model for use on CIFAR-10 data, following Chen
et al.  Specifically, it has a 3x3 input convolutional layer and no first 
maxpooling layer

SimCLR.py
--generates the SimCLR module.  specifically adds the projection head to 
ResNet50 and defines a forward command for processing pairs of samples at a 
time

train_util.py
--assorted utilities, in particular the transformation which returns a pair
of random transofrms to the same sample, the NT_Xent loss function and the 
LARS optimizer for training

ipynb files:

SimClr.ipynb 
-- Trains the self supervised model, allows for viewing of sample pairs of
images
-- allows for semisupervised learning using a given percentage of the CIFAR-10 
training data
-- allows for transfer learning by training the classifier on top of CIFAR-100