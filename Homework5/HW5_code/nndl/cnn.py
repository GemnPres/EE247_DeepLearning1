import numpy as np

from nndl.layers import *
from nndl.conv_layers import *
from utils.fast_layers import *
from nndl.layer_utils import *
from nndl.conv_layer_utils import *

import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize the weights and biases of a three layer CNN. To initialize:
    #     - the biases should be initialized to zeros.
    #     - the weights should be initialized to a matrix with entries
    #         drawn from a Gaussian distribution with zero mean and 
    #         standard deviation given by weight_scale.
    # ================================================================ #
    pad=1 
    stride=1
    std = weight_scale
    num_channel = input_dim[0]
    w_in = input_dim[1]
    self.params['W1'] = std * np.random.randn(num_filters, num_channel, filter_size, filter_size)
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = std * np.random.randn(num_filters, num_filters, filter_size, filter_size)
    self.params['b2'] = np.zeros(num_filters)
    self.bn_param_conv1 = {}
    self.bn_param_conv2 = {}
    self.bn_param_fc = {}

    stride = 1
    pad = int((filter_size - 1) / 2)

    """    
    w_conv_out = (w_in+2*pad-filter_size)/stride) + 1) 
    = (w_in+(filter_size - 1)- filter_size)/stride + 1
    = (w_in-1)/stride + 1
    
    -if stide==1, w_conv_out = w_in

    w_mp = (w_in-pool_width)/stride) + 1) 

    -if stide==2, pool_width==2, w_mp = w_in/2
    """

    w_conv = int((w_in-1)/stride + 1)
    w_mp = int(w_conv/2)
    h_mp = w_mp

    # output after mp: N*num_filters*h_mp*w_mp -> vectorize into N*(num_filters*h_mp*w_mp) -> FC
    # FC1: N*hidden_dim
    self.params['W3'] = std * np.random.randn(num_filters*h_mp*w_mp, hidden_dim)
    self.params['b3'] = np.zeros(hidden_dim)
    self.params['W4'] = std * np.random.randn(hidden_dim, num_classes)
    self.params['b4'] = np.zeros(num_classes)

    if self.use_batchnorm == True:
      self.bn_param_conv1 = {'mode': 'train'} 
      self.bn_param_conv2 = {'mode': 'train'} 
      self.bn_param_fc = {'mode': 'train'} 

      self.params['gamma1'] = np.ones(num_filters)
      self.params['beta1'] = np.zeros(num_filters)
      self.params['gamma2'] = np.ones(num_filters)
      self.params['beta2'] = np.zeros(num_filters)
      self.params['gamma3'] = np.ones(hidden_dim)
      self.params['beta3'] = np.zeros(hidden_dim)

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']

    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    """    
    w_conv = (w_in+2*pad-filter_size)/stride) + 1) 
    = (w_in+(filter_size - 1)- filter_size)/stride + 1
    = (w_in-1)/stride + 1

    if stide==1, w_conv = w_in
    """

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    """    
    w_mp = (w_in-pool_width)/stride) + 1) 

    if stide==2, pool_width==2
    w_mp = w_in/2
    """
    scores = None
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of the three layer CNN.  Store the output
    #   scores as the variable "scores".
    # ================================================================ #

    # BN performs different between training and testing
    mode = 'test' if y is None else 'train' 
    if self.use_batchnorm==True:
      self.bn_param_conv1["mode"] = mode
      self.bn_param_conv2["mode"] = mode
      self.bn_param_fc["mode"] = mode


    # conv same - BN - relu - conv same - BN - relu - 2x2 max pool - affine - BN - relu - affine - softmax
    batch_size = X.shape[0]

    conv1_out, cache_conv1 = conv_forward_fast(X, W1, b1, conv_param)

    if self.use_batchnorm == True: 
      gamma1, beta1 = self.params['gamma1'], self.params['beta1']
      bn_conv1, cache_bn_conv1 = spatial_batchnorm_forward(conv1_out, gamma1, beta1, self.bn_param_conv1)
      conv1_out = bn_conv1

    conv_ReLU1, cache_conv_ReLU1 = relu_forward(conv1_out)
    conv2_out, cache_conv2 = conv_forward_fast(conv_ReLU1, W2, b2, conv_param)

    if self.use_batchnorm == True: 
      gamma2, beta2 = self.params['gamma2'], self.params['beta2']
      bn_conv2, cache_bn_conv2 = spatial_batchnorm_forward(conv2_out, gamma2, beta2, self.bn_param_conv2)
      conv2_out = bn_conv2

    conv_ReLU2, cache_conv_ReLU2 = relu_forward(conv2_out)
    cnn_out, cache_cnn = max_pool_forward_fast(conv_ReLU2, pool_param)
    cnn_vectorize = np.reshape(cnn_out, [batch_size, -1])      

    fc1_out, cache_fc1 = affine_forward(cnn_vectorize, W3, b3)

    if self.use_batchnorm == True: 
      gamma3, beta3 = self.params['gamma3'], self.params['beta3']
      bn_fc, cache_bn_fc = batchnorm_forward(fc1_out, gamma3, beta3, self.bn_param_fc)
      fc1_out = bn_fc

    fc1_ReLU, cache_fc1_ReLU = relu_forward(fc1_out)
  
    fc2_out, cache_fc2 = affine_forward(fc1_ReLU, W4, b4)
    scores = fc2_out
    
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    # If in test mode:
    if y is None:
      return scores
    
    loss, grads = 0, {}
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass of the three layer CNN.  Store the grads
    #   in the grads dictionary, exactly as before (i.e., the gradient of 
    #   self.params[k] will be grads[k]).  Store the loss as "loss", and
    #   don't forget to add regularization on ALL weight matrices.
    # ================================================================ #

    # sf_loss - softmax - affine - relu - BN - affine - 2x2 max pool - relu - BN - conv same - conv same
    
    sf_loss, grad_softmax = softmax_loss(scores, y)   # y as true labels
    reg_loss = 0.5*self.reg*(np.linalg.norm(W1)**2 \
                             + np.linalg.norm(W2)**2\
                             + np.linalg.norm(W3)**2\
                             + np.linalg.norm(W4)**2)

    loss = sf_loss + reg_loss

    """ Forward Pass
    # conv same - BN - relu - conv same - BN - relu - 2x2 max pool - affine - BN - relu - affine - softmax
    batch_size = X.shape[0]

    conv1_out, cache_conv1 = conv_forward_fast(X, W1, b1, conv_param)

    if self.use_batchnorm == True: 
      gamma1, beta1 = self.params['gamma1'], self.params['beta1']
      bn_conv1, cache_bn_conv1 = spatial_batchnorm_forward(conv1_out, gamma1, beta1, self.bn_param_conv1)
      conv1_out = bn_conv1

    conv_ReLU1, cache_conv_ReLU1 = relu_forward(conv1_out)
    conv2_out, cache_conv2 = conv_forward_fast(conv_ReLU1, W2, b2, conv_param)

    if self.use_batchnorm == True: 
      gamma2, beta2 = self.params['gamma2'], self.params['beta2']
      bn_conv2, cache_bn_conv2 = spatial_batchnorm_forward(conv2_out, gamma2, beta2, self.bn_param_conv2)
      conv2_out = bn_conv2

    conv_ReLU2, cache_conv_ReLU2 = relu_forward(conv2_out)
    cnn_out, cache_cnn = max_pool_forward_fast(conv_ReLU2, pool_param)
    cnn_vectorize = np.reshape(cnn_out, [batch_size, -1])      

    fc1_out, cache_fc1 = affine_forward(cnn_vectorize, W3, b3)

    if self.use_batchnorm == True: 
      gamma3, beta3 = self.params['gamma3'], self.params['beta3']
      bn_fc, cache_bn_fc = batchnorm_forward(fc1_out, gamma3, beta3, self.bn_param_fc)
      fc1_out = bn_fc

    fc1_ReLU, cache_fc1_ReLU = relu_forward(fc1_out)
  
    fc2_out, cache_fc2 = affine_forward(fc1_ReLU, W4, b4)
    scores = fc2_out
    """

    grad_fc1_ReLU, grad_W4, grad_b4 = affine_backward(grad_softmax, cache_fc2)
    grad_fc1_out = relu_backward(grad_fc1_ReLU, cache_fc1_ReLU)
    
    if self.use_batchnorm == True: 
      grad_bn_fc, grad_gamma3, grad_beta3 = batchnorm_backward(grad_fc1_out, cache_bn_fc)
      grad_fc1_out = grad_bn_fc

    grad_cnn_vectorize, grad_W3, grad_b3 = affine_backward(grad_fc1_out, cache_fc1)
    
    # grad_cnn_vectorize shape: (N, f*h_mp*w_mp)
    grad_cnn = np.reshape(grad_cnn_vectorize, cnn_out.shape)
    grad_conv_ReLU2 = max_pool_backward_fast(grad_cnn, cache_cnn)
    grad_conv_out2 = relu_backward(grad_conv_ReLU2, cache_conv_ReLU2)

    if self.use_batchnorm == True: 
      grad_bn_conv2, grad_gamma2, grad_beta2 = spatial_batchnorm_backward(grad_conv_out2, \
                                                                          cache_bn_conv2)
      grad_conv_out2 = grad_bn_conv2

    grad_conv_ReLU1, grad_W2, grad_b2 = conv_backward_fast(grad_conv_out2, cache_conv2)
    grad_conv_out1 = relu_backward(grad_conv_ReLU1, cache_conv_ReLU1)

    if self.use_batchnorm == True: 
      grad_bn_conv1, grad_gamma1, grad_beta1 = spatial_batchnorm_backward(grad_conv_out1, \
                                                                          cache_bn_conv1)
      grad_conv_out1 = grad_bn_conv1

    grad_X, grad_W1, grad_b1 = conv_backward_fast(grad_conv_out1, cache_conv1)


    grad_W1_reg = 0.5*self.reg*2*W1
    grad_W2_reg = 0.5*self.reg*2*W2
    grad_W3_reg = 0.5*self.reg*2*W3
    grad_W4_reg = 0.5*self.reg*2*W4

    grads['W4'] = grad_W4 + grad_W4_reg
    grads['b4'] = grad_b4
    grads['W3'] = grad_W3 + grad_W3_reg
    grads['b3'] = grad_b3
    grads['W2'] = grad_W2 + grad_W2_reg
    grads['b2'] = grad_b2
    grads['W1'] = grad_W1 + grad_W1_reg
    grads['b1'] = grad_b1

    if self.use_batchnorm == True:
      grads['gamma3'] = grad_gamma3
      grads['beta3'] = grad_beta3
      grads['gamma2'] = grad_gamma2
      grads['beta2'] = grad_beta2
      grads['gamma1'] = grad_gamma1
      grads['beta1'] = grad_beta1

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return loss, grads