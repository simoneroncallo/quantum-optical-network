# This module prepares the dataset to train the quantum optical network

import numpy as np
from keras import datasets

def rgb2gray(rgb):
  """ Convert an RGB digital image to greyscale. """
  return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def dataset_preparation(dataset,numTrainImgs):
  # Choose the dataset
  if dataset == 'mnist':
    (trainImgs, trainLabels), (testImgs, testLabels) =\
     datasets.mnist.load_data()

    # Filter 0 and 1 from the dataset
    train0s, test0s = np.where(trainLabels == 0), np.where(testLabels == 0)
    train1s, test1s = np.where(trainLabels == 1), np.where(testLabels == 1)

  elif dataset == 'fashion':
    (trainImgs, trainLabels), (testImgs, testLabels) =\
      datasets.fashion_mnist.load_data()

    # Filter 0 and 1 from the dataset
    train0s, test0s = np.where(trainLabels == 0), np.where(testLabels == 0)
    train1s, test1s = np.where(trainLabels == 2), np.where(testLabels == 2)

  elif dataset == 'cifar':
    # Classes 0: Plane | 3: Cat | 5: Dog
    (trainImgs, trainLabels), (testImgs, testLabels) =\
     datasets.cifar10.load_data()

    # Filter 0 and 1 from the dataset
    train0s, test0s = np.where(trainLabels == 0), np.where(testLabels == 0)
    train1s, test1s = np.where(trainLabels == 5), np.where(testLabels == 5)

  else:
    raise ValueError('Dataset not available')

  train0sImgs = trainImgs[train0s[0]]
  train1sImgs = trainImgs[train1s[0]] # Raw dataset
  # train1sImgs = trainImgs[train1s[0]][0:train0sImgs.shape[0]] # Balanced dataset

  test0sImgs = testImgs[test0s[0]]
  test1sImgs = testImgs[test1s[0]]

  trainImgs = np.concatenate((train0sImgs, train1sImgs), axis = 0)
  testImgs = np.concatenate((test0sImgs, test1sImgs), axis = 0)

  # Create the dataset of images and labels (0s and 1s)
  train0Labels = np.zeros(train0sImgs.shape[0])
  train1Labels = np.ones(train1sImgs.shape[0]) # Raw dataset
  # train1Labels = np.ones(train0sImgs.shape[0]) # Balanced dataset
  trainLabels = np.concatenate((train0Labels, train1Labels), axis = 0)

  test0Labels = np.zeros(test0sImgs.shape[0])
  test1Labels = np.ones(test1sImgs.shape[0])
  testLabels = np.concatenate((test0Labels, test1Labels), axis = 0)

  # Reshuffle images and labels consistently
  idxs = np.arange(trainImgs.shape[0])
  np.random.shuffle(idxs)

  trainImgs = trainImgs[idxs]
  trainLabels = trainLabels[idxs]

  # Convert to float
  trainImgs = trainImgs.astype(np.float64)
  testImgs = testImgs.astype(np.float64)
  trainLabels = trainLabels.astype(np.float64)
  testLabels = testLabels.astype(np.float64)

  if dataset == 'mnist' or dataset == 'fashion':
    # Padding from 28x28 to 32x32
    trainImgs = np.pad(trainImgs, ((0,0),(2,2),(2,2)), mode='constant', \
                      constant_values = 0)
    testImgs = np.pad(testImgs, ((0,0),(2,2),(2,2)), mode='constant', \
                      constant_values = 0)

  elif dataset == 'cifar':
    # Greyscale conversion
    trainImgs = rgb2gray(trainImgs)
    testImgs = rgb2gray(testImgs)

  # Reduce the training set
  trainImgs = trainImgs[:numTrainImgs,:,:]
  trainLabels = trainLabels[:numTrainImgs]

  # Vectorized the dataset
  trainImgs = trainImgs.reshape(trainImgs.shape[0],-1)
  testImgs = testImgs.reshape(testImgs.shape[0],-1)

  print('Training set has shape', trainImgs.shape)
  print('Test set has shape', testImgs.shape, '\n')

  # # Phases encoding
  # # Normalize white to 1
  # trainImgs /= 255
  # testImgs /= 255
  # trainImgs = np.exp(1j*np.pi*trainImgs[:,:])
  # testImgs = np.exp(1j*np.pi*testImgs[:,:])

  # Normalization
  trainImgs_norms = np.sqrt(np.sum(np.square(np.abs(trainImgs)), axis = 1))
  trainImgs /= trainImgs_norms[:, np.newaxis]

  testImgs_norms = np.sqrt(np.sum(np.square(np.abs(testImgs)), axis = 1))
  testImgs /= testImgs_norms[:, np.newaxis]

  return trainImgs, trainLabels, testImgs, testLabels