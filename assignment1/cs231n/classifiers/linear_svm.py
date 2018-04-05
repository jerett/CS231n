import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape)  # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1  # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, y[i]] -= X[i]
        dW[:, j] += X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength
  """
  loss = 0.0
  # DxC
  dW = np.zeros(W.shape)  # initialize the gradient as zero

  #############################################################################
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  num_train = X.shape[0]
  # NxC, each row is train data each class score
  score = X.dot(W)
  # NxC
  margins = np.maximum(0, (score.T - score[range(num_train), y]).T + 1)
  # set all correct label score 1
  # print(margins[0])
  margins[range(num_train), y] = 0
  loss = np.sum(margins) / num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  # sclar, (N,)
  margins_count = np.sum(margins > 0, axis=1)
  # D*C
  X_plus = X.T.dot(margins > 0)
  # N*C
  X_minus_mask = np.zeros(margins.shape)
  X_minus_mask[range(num_train), y] = margins_count
  # D*C
  X_minus = X.T.dot(X_minus_mask)
  dW += X_plus
  dW -= X_minus
  dW /= num_train
  dW += reg * W
  # this is a for-loop solution
  # it = np.nditer(margins, flags=['multi_index'])
  # while not it.finished:
  #   sample_index = it.multi_index[0]
  #   claz = it.multi_index[1]
  #   label = y[sample_index]
  #   sample = X[sample_index]
  #   # print(claz, label)
  #   # print(it.multi_index[0], it.multi_index[1])
  #   if it[0] > 0 and claz != label:
  #     dW[:, label] -= sample
  #     dW[:, claz] += sample
  #   it.iternext()
  # dW /= num_train


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
