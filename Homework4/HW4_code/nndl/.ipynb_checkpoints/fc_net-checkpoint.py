import numpy as np
import pdb

from .layers import *
from .layer_utils import *

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dims=100, num_classes=10,
               dropout=0, weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dims: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize W1, W2, b1, and b2.  Store these as self.params['W1'], 
    #   self.params['W2'], self.params['b1'] and self.params['b2']. The
    #   biases are initialized to zero and the weights are initialized
    #   so that each parameter has mean 0 and standard deviation weight_scale.
    #   The dimensions of W1 should be (input_dim, hidden_dim) and the
    #   dimensions of W2 should be (hidden_dims, num_classes)
    # ================================================================ #
    
    std = weight_scale
    self.params['W1'] = std * np.random.randn(input_dim, hidden_dims)
    self.params['b1'] = np.zeros(hidden_dims)
    self.params['W2'] = std * np.random.randn(hidden_dims, num_classes)
    self.params['b2'] = np.zeros(num_classes)

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of the two-layer neural network. Store
    #   the class scores as the variable 'scores'.  Be sure to use the layers
    #   you prior implemented.
    # ================================================================ #    

    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']

    layer1, layer1_cache = affine_forward(X, W1, b1)
    layer1_bn, layer_cache_bn = batchnorm_forward(layer1, gamma, beta, bn_param)
    ReLU_act, ReLU_act_cache = relu_forward(layer1)
    scores, layer2_cache = affine_forward(ReLU_act, W2, b2)
      
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    
    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass of the two-layer neural net.  Store
    #   the loss as the variable 'loss' and store the gradients in the 
    #   'grads' dictionary.  For the grads dictionary, grads['W1'] holds
    #   the gradient for W1, grads['b1'] holds the gradient for b1, etc.
    #   i.e., grads[k] holds the gradient for self.params[k].
    #
    #   Add L2 regularization, where there is an added cost 0.5*self.reg*W^2
    #   for each W.  Be sure to include the 0.5 multiplying factor to 
    #   match our implementation.
    #
    #   And be sure to use the layers you prior implemented.
    # ================================================================ #    

    sf_loss, grad_softmax = softmax_loss(scores, y)
    reg_loss = 0.5*self.reg*(np.linalg.norm(W1)**2 + np.linalg.norm(W2)**2)
    loss = sf_loss + reg_loss

    """ forward pass:
    layer1, layer1_cache = affine_forward(X, W1, b1)
    ReLU_act, ReLU_act_cache = relu_forward(layer1)
    scores, layer2_cache = affine_forward(ReLU_act, W2, b2)    
    """
    
    grad_ReLU, grad_W2, grad_b2 = affine_backward(grad_softmax, layer2_cache)
    grad_layer1 = relu_backward(grad_ReLU, ReLU_act_cache)
    _, grad_W1, grad_b1 = affine_backward(grad_layer1, layer1_cache)

    grad_W1_reg = 0.5*self.reg*2*W1
    grad_W2_reg = 0.5*self.reg*2*W2
    
    grads['W2'] = grad_W2 + grad_W2_reg
    grads['b2'] = grad_b2
    grads['W1'] = grad_W1 + grad_W1_reg
    grads['b1'] = grad_b1
      
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    
    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize all parameters of the network in the self.params dictionary.
    #   The weights and biases of layer 1 are W1 and b1; and in general the 
    #   weights and biases of layer i are Wi and bi. The
    #   biases are initialized to zero and the weights are initialized
    #   so that each parameter has mean 0 and standard deviation weight_scale.
    #
    #   BATCHNORM: Initialize the gammas of each layer to 1 and the beta
    #   parameters to zero.  The gamma and beta parameters for layer 1 should
    #   be self.params['gamma1'] and self.params['beta1'].  For layer 2, they
    #   should be gamma2 and beta2, etc. Only use batchnorm if self.use_batchnorm 
    #   is true and DO NOT do batch normalize the output scores.
    # ================================================================ #
    
    dim_list = np.hstack([input_dim, hidden_dims, num_classes])
    std = weight_scale
    
    for i in range(self.num_layers):
        key_W = "W" + str(i+1)
        key_b = "b" + str(i+1)
        self.params[key_W] = std * np.random.randn(dim_list[i], dim_list[i+1])
        self.params[key_b] = np.zeros(dim_list[i+1])

        # no BN gamma beta for the output layer
        if self.use_batchnorm == True:
            if i < self.num_layers-1:
                key_gamma = "gamma" + str(i+1)
                key_beta = "beta" + str(i+1)
                self.params[key_gamma] = np.ones(dim_list[i+1])
                self.params[key_beta] = np.zeros(dim_list[i+1])

      
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    
    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
        self.dropout_param = {'mode': 'train', 'p': dropout}
    if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
        self.bn_params = [{'mode': 'train'} for i in np.arange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.items():
        self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'
    
    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
        self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
        for bn_param in self.bn_params:
            bn_param[mode] = mode

    scores = None
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of the FC net and store the output
    #   scores as the variable "scores".
    #
    #   BATCHNORM: If self.use_batchnorm is true, insert a bathnorm layer
    #   between the affine_forward and relu_forward layers.  You may
    #   also write an affine_batchnorm_relu() function in layer_utils.py.
    #
    #   DROPOUT: If dropout is non-zero, insert a dropout layer after
    #   every ReLU layer.
    # ================================================================ #
    
    #  Add  DROPOUT: If dropout is non-zero, insert a dropout layer after
    
    cache_ls = []
    in_ = X


    """ FORWARD with BN (+ Dropout)
    layer1, layer1_cache = affine_forward(X, W1, b1)
    cache_ls.append(layer1_cache)

    bn_layer1, bn_layer1_cache =batchnorm_forward(layer1, gamma1, beta1, bn_param[0])
    
    ReLU_act, ReLU_act_cache = relu_forward(bn_layer)
    cache_ls.append(ReLU_act_cache)

    (drop_layer1, drop_layer1_cache = dropout_forward(ReLU_act, dropout_param)
    cache_ls.append(ReLU_act_cache)                                          )
    
    layer2, layer2_cache = affine_forward(ReLU_act (or drop_layer1) , W2, b2)
    cache_ls.append(layerN_cache)

    bn_layer2, bn_layer2_cache =batchnorm_forward(layer2, gamma2, beta2, bn_param)

    ReLU_act, ReLU_act_cache2 = relu_forward(layer2)
    cache_ls.append(ReLU_act2)

    (drop_layer2, drop_layer2_cache = dropout_forward(ReLU_act, dropout_param)
    cache_ls.append(drop_layer2_cache)                                       )
    
    scores, layer3_cache = affine_forward(ReLU_act2 (or drop_layer2), W3, b3)
    cache_ls2.append(layer3_cache)
    """
 
    for i in range(self.num_layers):
        if i < self.num_layers-1:
            key_W = "W" + str(i+1)
            key_b = "b" + str(i+1)
            W_i, b_i = self.params[key_W], self.params[key_b]
            
            if self.use_batchnorm == True:  
                key_gamma = "gamma" + str(i+1)
                key_beta = "beta" + str(i+1)
                gamma_i, beta_i = self.params[key_gamma], self.params[key_beta]

            # stack up affine-batchnorm-ReLU-Dropout layers for #(num_layers-1) times 
            # Consider if BN is used or not; consider if Dropout is used or not
            layer_out, layer_cache = affine_forward(in_, W_i, b_i)
            cache_ls.append(layer_cache)

            if self.use_batchnorm == True:  
                layer_out, bn_layer_cache =batchnorm_forward(layer_out, gamma_i, beta_i, self.bn_params[i])
                cache_ls.append(bn_layer_cache)
            
            ReLU_act, ReLU_act_cache = relu_forward(layer_out)
            cache_ls.append(ReLU_act_cache)
            block_out = ReLU_act
            
            if self.use_dropout == True:
                drop_layer, drop_layer_cache = dropout_forward(ReLU_act, self.dropout_param)
                cache_ls.append(drop_layer_cache)
                block_out = drop_layer

            in_ = block_out

        # in case used BatchNorm or Dropout, avoid BN, ReLU, and Dropout at the output layer
        else:
            key_W = "W" + str(i+1)
            key_b = "b" + str(i+1)
            W_i, b_i = self.params[key_W], self.params[key_b]
            layer_out, layer_cache = affine_forward(in_, W_i, b_i)
            cache_ls.append(layer_cache)
            scores = layer_out

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    
    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backwards pass of the FC net and store the gradients
    #   in the grads dict, so that grads[k] is the gradient of self.params[k]
    #   Be sure your L2 regularization includes a 0.5 factor.
    #
    #   BATCHNORM: Incorporate the backward pass of the batchnorm.
    #
    #   DROPOUT: Incorporate the backward pass of dropout.
    # ================================================================ #

    sf_loss, grad_softmax = softmax_loss(scores, y)
    reg_loss = 0.0

    # Calc Regularization term
    for i in range(self.num_layers):
        key_W = "W" + str(i+1)
        W_i = self.params[key_W]
        reg_loss += 0.5*self.reg*(np.linalg.norm(W_i)**2)
        
    loss = sf_loss + reg_loss
      
    """BACKWARD with BN
    grad_ReLU2, grad_W3, grad_b3 = affine_backward(grad_softmax, layer_cache[6])  #i==7
    grad_layer2 = relu_backward(grad_ReLU2, layer_cache[5])  #i==6
    grad_layer2_bn = batchnorm_backward(grad_layer2, layer_cache[4])  #i==5
    grad_ReLU1, grad_W2, grad_b2 = affine_backward(grad_layer2_bn, layer_cache[3])   #i==4
    grad_layer1 = relu_backward(grad_ReLU1, layer_cache[2])  #i==3
    grad_layer1_bn = batchnorm_backward(grad_layer1, layer_cache[1])  #i==2
    _, grad_W1, grad_b1 = affine_backward(grad_layer1, layer_cache[0])   #i==1
    """

    """BACKWARD with Dropout
    grad_blockout2, grad_W3, grad_b3 = affine_backward(grad_softmax, layer_cache[6])  #i==7
    grad_ReLU2 = dropout_backward(grad_blockout2, layer_cache[5])  #i==6
    grad_layer2 = relu_backward(grad_ReLU2, layer_cache[5])  #i==5
    grad_blockout1, grad_W2, grad_b2 = affine_backward(grad_layer2, layer_cache[3])   #i==4
    grad_ReLU1 = dropout_backward(grad_blockout1, layer_cache[2])  #i==3
    grad_layer1 = relu_backward(grad_ReLU1, layer_cache[1)  #i==2
    _, grad_W1, grad_b1 = affine_backward(grad_layer1, layer_cache[0])   #i==1
    """

    """BACKWARD with BN + Dropout
    grad_blockout2, grad_W3, grad_b3 = affine_backward(grad_softmax, layer_cache[8])  #i==9
    grad_ReLU2 = dropout_backward(grad_blockout2, layer_cache[7])  #i==8
    grad_layer2 = relu_backward(grad_ReLU2, layer_cache[6])  #i==7
    grad_layer2_bn = batchnorm_backward(grad_layer2, layer_cache[5])  #i==6
    grad_blockout1, grad_W2, grad_b2 = affine_backward(grad_layer2_bn, layer_cache[4])   #i==5
    grad_ReLU1 = dropout_backward(grad_blockout1, layer_cache[3])  #i==4
    grad_layer1 = relu_backward(grad_ReLU1, layer_cache[2])  #i==3
    grad_layer1_bn = batchnorm_backward(grad_layer1, layer_cache[1])  #i==2
    _, grad_W1, grad_b1 = affine_backward(grad_layer1, layer_cache[0])   #i==1
    """
    
    # len(cache_ls) is guaranteed to be an odd num due to sandwiching (no BN & activation @ output layer)
    # start with W_i (W3) and b_i (b3) corresponding with the output layer; i==7
    grad_backnode, grad_W_i, grad_b_i = affine_backward(grad_softmax, cache_ls[-1])
    key_W = "W" + str(self.num_layers)
    key_b = "b" + str(self.num_layers)
    grad_Wi_reg = 0.5*self.reg*2*self.params[key_W]

    grads[key_W] = grad_W_i+grad_Wi_reg
    grads[key_b] = grad_b_i
    
    # i: 6,3; len(cache_ls)=7 with BN given 2 hidden layers
    # start with cache[5]; update for W2,b2,gamma2,beta2,W1,b1,gamma1,beta1
    # BN has a sandwich of 3 blocks: Affine->BN->ReLU
    if self.use_batchnorm == True and self.use_dropout == False:  
        for i in range(len(cache_ls)-1, 1, -3):
            grad_bn = relu_backward(grad_backnode, cache_ls[i-1]) 
            grad_layer, grad_gamma_i, grad_beta_i = batchnorm_backward(grad_bn, cache_ls[i-2]) 
            grad_backnode, grad_W_i, grad_b_i = affine_backward(grad_layer, cache_ls[i-3])
            key_W = "W" + str(i//3)
            key_b = "b" + str(i//3)
            key_gamma = "gamma" + str(i//3)
            key_beta = "beta" + str(i//3)
            grad_Wi_reg = 0.5*self.reg*2*self.params[key_W]
    
            grads[key_W] = grad_W_i + grad_Wi_reg
            grads[key_b] = grad_b_i
            grads[key_gamma] = grad_gamma_i
            grads[key_beta] = grad_beta_i

    # i: 6,3; len(cache_ls)=7 with Dropout given 2 hidden layers
    # start with cache[5]; update for W2,b2,W1,b1
    # BN has a sandwich of 3 blocks: Affine->ReLU->Dropout
    elif self.use_batchnorm == False and self.use_dropout == True:  
        for i in range(len(cache_ls)-1, 1, -3):
            grad_ReLU = dropout_backward(grad_backnode, cache_ls[i-1])
            grad_layer = relu_backward(grad_ReLU, cache_ls[i-2]) 
            grad_backnode, grad_W_i, grad_b_i = affine_backward(grad_layer, cache_ls[i-3])
            key_W = "W" + str(i//3)
            key_b = "b" + str(i//3)
            grad_Wi_reg = 0.5*self.reg*2*self.params[key_W]
            grads[key_W] = grad_W_i + grad_Wi_reg
            grads[key_b] = grad_b_i
    
    # i: 8,4; len(cache_ls)=9 with BN & Dropout each given 2 hidden layers
    # start with cache[7]; update for W2,b2,gamma2,beta2,W1,b1,gamma1,beta1
    # BN has a sandwich of 4 blocks: Affine->BN->ReLU->Dropout
    elif self.use_batchnorm == True and self.use_dropout == True:  
        for i in range(len(cache_ls)-1, 1, -4):
            grad_ReLU = dropout_backward(grad_backnode, cache_ls[i-1])
            grad_bn = relu_backward(grad_ReLU, cache_ls[i-2]) 
            grad_layer, grad_gamma_i, grad_beta_i = batchnorm_backward(grad_bn, cache_ls[i-3]) 
            grad_backnode, grad_W_i, grad_b_i = affine_backward(grad_layer, cache_ls[i-4])
            key_W = "W" + str(i//4)
            key_b = "b" + str(i//4)
            key_gamma = "gamma" + str(i//4)
            key_beta = "beta" + str(i//4)
            grad_Wi_reg = 0.5*self.reg*2*self.params[key_W]
    
            grads[key_W] = grad_W_i + grad_Wi_reg
            grads[key_b] = grad_b_i
            grads[key_gamma] = grad_gamma_i
            grads[key_beta] = grad_beta_i
    
    # i: 4,2; len(cache_ls)=5 with No BN and No Dropout
    # start with cache[4]; update for W2,b2,W1,b1
    # Without BN has a sandwich of 2 blocks: Affine->ReLU
    else:
        for i in range(len(cache_ls)-1, 1, -2):
            grad_layer = relu_backward(grad_backnode, cache_ls[i-1])        
            grad_backnode, grad_W_i, grad_b_i = affine_backward(grad_layer, cache_ls[i-2])
            key_W = "W" + str(i//2)
            key_b = "b" + str(i//2)
            grad_Wi_reg = 0.5*self.reg*2*self.params[key_W]
            grads[key_W] = grad_W_i + grad_Wi_reg
            grads[key_b] = grad_b_i

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    
    return loss, grads
