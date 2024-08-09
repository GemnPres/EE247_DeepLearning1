import numpy as np
from nndl.layers import *
import pdb
from tqdm import tqdm

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #
  
  w_in, h_in = x.shape[3], x.shape[2]
  w_kernel, h_kernel = w.shape[3], w.shape[2]
  num_img, num_channel, num_kernel = x.shape[0], x.shape[1], w.shape[0]

  w_out = int(np.floor((w_in+2*pad-w_kernel)/stride) + 1)   
  h_out = int(np.floor((h_in+2*pad-h_kernel)/stride) + 1)   
  out = np.zeros([num_img, num_kernel, h_out, w_out]) 

  """
  for every img input
  iterate thru ouput pixels [i,j] ([vertical axis, horizontal axis]) 
  for each pixel in output, see the corresponding input window
  multiply input win with the filter(weight), take a sum over # channel
  conv(1*receptive_field*#ch, #filter*filter_size(==receptive_field)*#ch) -> #filter*1 (1 #filter-dim bar for [i,j] pixel)
  that's how we do for ONE output pixel [i,j] for ONE img example n 
  """

  zp_row = np.zeros([num_img, num_channel, pad, w_in])
  zp_col = np.zeros([num_img, num_channel, h_in+2*pad, pad])
  # use np.concat to pad feature maps (NOT np.stack)
  xpad = np.concatenate([zp_col, np.concatenate([zp_row, x, zp_row], axis=2), zp_col], axis=3)  
  
  for n in range(num_img):
    for i in range(h_out):                      
      for j in range(w_out):
        xpad_n = xpad[n]  # xpad_n of shape [#ch, h_in, w_in]

        # determine the range of the input that is multiplied by the filters
        x_seg = xpad_n[:, 0+i*stride:0+i*stride+h_kernel, 0+j*stride:0+j*stride+w_kernel]  
               
        # calculate the conv output; b is bias term
        # segment of the input * filter + b; compress over channel dim, w dim, and h dim 
        # expected shape: [#filter, 1, 1]
        out[n,:,i,j] = np.sum(np.multiply(x_seg, w), axis=(1,2,3)) + b

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
    
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  N, F, out_height, out_width = dout.shape
  x, w, b, conv_param = cache
  
  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #

  dx, dw, db = np.zeros(x.shape), np.zeros(w.shape), np.zeros(b.shape)

  w_in, h_in = x.shape[3], x.shape[2]
  w_kernel, h_kernel = w.shape[3], w.shape[2]
  num_img, num_channel, num_kernel = x.shape[0], x.shape[1], w.shape[0]

  w_out = int(np.floor((w_in+2*pad-w_kernel)/stride) + 1)   
  h_out = int(np.floor((h_in+2*pad-h_kernel)/stride) + 1)   
  
  # xpad shape: [N, C, h_in+zp, w_in+zp]
  # w kernel shape: [F, C, h_kernel, w_kernel]
  # out shape: [N, F, h_out, w_out]
  # b shape: [F]

  # Calculate dw with numerical derivation
  for n in tqdm(range(num_img)):
    for f in range(num_filts):
      for c in range(num_channel):
        for i in range(h_out):
          for j in range(w_out):
            dout_n_f = dout[n][f]   # dout_n_f: matrix of shape [h_out, w_out]; work with 1 layer 
            x_seg = xpad[n, c, 0+i*stride:0+i*stride+h_kernel, 0+j*stride:0+j*stride+w_kernel]  # x_seg matrix of shape [h_kernel, w_kernel]

            # find the corresponding receptive field x_seg in x_pad during conv (same shape as kernel), local grad is x_seg
            # since during conv, each receptive field contributes to kernel, need to accumulate grad with +=
            # output of conv is scalar y[i,j], thus upperstream grad is scalar
            dw[f, c, :, :] += x_seg*dout_n_f[i][j]


  # Calculate dx with numerical derivation
  # get a container for dx_pad since we do math on x_pad
  # the indexing in conv forward pass still applies; remember to trim dx_pad->dx
  dx_pad = np.zeros(xpad.shape)

  for n in tqdm(range(num_img)):
    for f in range(num_filts):
      for c in range(num_channel):
        for i in range(h_out):
          for j in range(w_out):
            dout_n_f = dout[n][f]   # dout_n_f: matrix of shape [h_out, w_out]; work with 1 layer
            kernel_f_c = w[f][c]  # kernel_f_c: matrix of shape [h_kernel, w_jernel]; work with 1 layer of filter

            # find the corresponding receptive field in x_pad during conv (same shape as kernel), local grad is the kernel matrix
            # since we have overlapping regions in x_pad during conv, need to accumulate grad with +=
            # output of conv is scalar y[i,j], thus upperstream grad is scalar
            dx_pad[n, c, 0+i*stride:0+i*stride+h_kernel, 0+j*stride:0+j*stride+w_kernel] += kernel_f_c*dout_n_f[i][j]

  dx = dx_pad[:,:,pad:-pad,pad:-pad]

  """
  Faster Method if dealing with conv 3*3 same

  # Calculate dw
  for n in tqdm(range(num_img)):
    for f in range(num_filts):
      for c in range(num_channel):
        for i in range(h_kernel):                      
          for j in range(w_kernel):
            
            # work with 1 layer of feature map (matrix) for 1 example
            xpad_n_c = xpad[n][c]  # xpad_n of shape [h_in, w_in], 

            # backprop w.r.t to W; dL/dW has a 2Dconv relationship with dout
            x_seg = xpad_n_c[0+i*stride:0+i*stride+h_out, 0+j*stride:0+j*stride+w_out]  

            # each of the nth example contributes to dw, thus accumulate dw
            # scalar value to assign; assign to 1 entry in the dw matrix
            dw[f, c, i, j] += np.sum(np.multiply(x_seg, dout[n][f]), axis=(0,1)) 
  
  # Calculate dx with conv 
  for n in tqdm(range(num_img)):
    for f in range(num_filts):
      for c in range(num_channel):
        for k in range(int(h_in)):                      
          for l in range(int(w_in)):
            
            dout_n_f = dout[n][f]   # dout_n_f: matrix of shape [h_out, w_out]; work with 1 layer
            zp_out_col = np.zeros([dout_n_f.shape[0], pad])
            zp_out_row = np.zeros([pad, 2*pad+dout_n_f.shape[1]])
            dout_n_f_zp = np.vstack([zp_out_row, np.hstack([zp_out_col, dout_n_f, zp_out_col]), zp_out_row])  # get padded dy
            kernel_f_c = w[f][c]
            kernel_flip = np.flip(kernel_f_c)  # get flipped kernel 
            h_flipkernel, w_flipkernel = kernel_flip.shape[0], kernel_flip.shape[1]

            # backprop w.r.t to X; dL/dX has a 2Dconv relationship with dout
            dout_n_f_zp_seg = dout_n_f_zp[0+k*stride:0+k*stride+h_flipkernel, 0+l*stride:0+l*stride+w_flipkernel]  

            # each of the nth example contributes to dx, thus accumulate dx
            # scalar value to assign; assign to 1 entry in the dx matrix
            
            dx[n, c, k, l] += np.sum(np.multiply(dout_n_f_zp_seg, kernel_flip), axis=(0,1)) 
  """

  # all entries have +b contribution, which correspond to local grad 1
  db = np.sum(dout, axis=(0,2,3))
  
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #

  num_img, num_channel, w_in, h_in = x.shape[0], x.shape[1], x.shape[3], x.shape[2]
  h_kernel, w_kernel, stride = pool_param["pool_height"], pool_param["pool_width"], pool_param["stride"]
  h_mp = int(np.floor((w_in-w_kernel)/stride) + 1)   
  w_mp = int(np.floor((h_in-h_kernel)/stride) + 1)   
  out = np.zeros([num_img, num_channel, h_mp, w_mp])   

  for n in range(num_img):
    for c in range(num_channel):
      for i in range(h_mp):                      
        for j in range(w_mp):
          feature_map = x[n][c]  # feature_map of shape [h_in, w_in]

          # determine the range of the input that we need to find the max
          # do it for every example and for every channel
          map_seg = feature_map[0+i*stride:0+i*stride+h_kernel, 0+j*stride:0+j*stride+w_kernel]  
          out[n,c,i,j] = np.amax(map_seg)
    
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #

  num_img, num_channel, w_in, h_in = x.shape[0], x.shape[1], x.shape[3], x.shape[2]
  h_kernel, w_kernel, stride = pool_param["pool_height"], pool_param["pool_width"], pool_param["stride"]
  h_mp = int(np.floor((w_in-w_kernel)/stride) + 1)   
  w_mp = int(np.floor((h_in-h_kernel)/stride) + 1)   
  dx = np.zeros(x.shape)   

  for n in tqdm(range(num_img)):
    for c in range(num_channel):
      for i in range(h_mp):                      
        for j in range(w_mp):
          feature_map = x[n][c]  # feature_map of shape [h_in, w_in]
          dout_n_c = dout[n][c]  # dout_n_c of shape [h_mp, w_mp]

          # determine the range of the input that we need to find the max for backprop
          # do it for every example and for every channel
          dx_seg = feature_map[0+i*stride:0+i*stride+h_kernel, 0+j*stride:0+j*stride+w_kernel]  # find the mp patch 
          max_loc_i, max_loc_j = np.unravel_index(np.argmax(dx_seg), dx_seg.shape)   # find max from the patch
          dx[n, c, max_loc_i+i*stride, max_loc_j+j*stride] = 1*dout_n_c[i][j]  # convert patch max coord to its correspondence on the feature map
                                                                               # pass grad like ReLU

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  """
  In CNN:
  Spatial batch-normalization is to reshape the (N, C, H, W) array into (N*H*W, C)
  and perform batch normalization on this array.

  In: (N, C, H, W) array
  Out: (N, C, H, W) array
  """

  # N, C, H, W = x.shape
  x_permute = np.transpose(x, [0,2,3,1])  # permute axes into shape (N, H, W, C)
  x_spatial_bn = np.reshape(x_permute, [np.prod(x_permute.shape[0:3]), x_permute.shape[-1]])  # reshape into shape (N*H*W, C)
  out_bn, cache = batchnorm_forward(x_spatial_bn, gamma, beta, bn_param)  # out_bn shape: (N*H*W, C)
  out_permute = np.reshape(out_bn, x_permute.shape)   # out_permute shape: (N, H, W, C)
  out = np.transpose(out_permute, [0,3,1,2])  # permute axes back into shape (N, C, H, W)

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
    
  dout_permute = np.transpose(dout, [0,2,3,1])  # permute axes into shape (N, H, W, C)
  dout_bn = np.reshape(dout_permute, [np.prod(dout_permute.shape[0:3]), dout_permute.shape[-1]])   # dout_bn shape: (N*H*W, C)
  dx_bn, dgamma, dbeta = batchnorm_backward(dout_bn, cache)   # dx_bn shape: (N*H*W, C)
  dx_bn_permute = np.reshape(dx_bn, dout_permute.shape)   # out_permute shape: (N, H, W, C)
  dx = np.transpose(dx_bn_permute, [0,3,1,2])  # permute axes back into shape (N, C, H, W)

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx, dgamma, dbeta