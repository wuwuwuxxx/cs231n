from cs231n.layers import *
from cs231n.fast_layers import *


def affine_relu_forward(x, w, b):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db


pass


def conv_relu_forward(x, w, b, conv_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache


def conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db

def conv_bn_relu_forward(x, w, gamma, beta, conv_param, bn_param):
  a, conv_cache = conv_forward_fast(x, w, conv_param)
  out, out_cache = sbn_relu_forward(a, gamma, beta, bn_param)
  cache = (conv_cache, out_cache)
  return out, cache

def conv_bn_relu_backward(dout, cache):
  conv_cache, out_cache = cache
  dx, dgamma, dbeta = sbn_relu_backward(dout, out_cache)
  dx, dw = conv_backward_fast(dx, conv_cache)
  return dx, dw, dgamma, dbeta

def sbn_relu_forward(x, gamma, beta, bn_param):
  s, bn_cache = spatial_batchnorm_forward(x, gamma, beta, bn_param)
  out, relu_cache = relu_forward(s)
  cache = (bn_cache, relu_cache)
  return out, cache

def sbn_relu_backward(dout, cache):
  bn_cache, relu_cache = cache
  dr = relu_backward(dout, relu_cache)
  dx, dgamma, dbeta = spatial_batchnorm_backward(dr, bn_cache)
  return dx, dgamma, dbeta

def bn_relu_forward(x, gamma, beta, bn_param):
  s, bn_cache = batchnorm_forward(x, gamma, beta, bn_param)
  out, relu_cache = relu_forward(s)
  cache = (bn_cache, relu_cache)
  return out, cache

def bn_relu_backward(dout, cache):
  bn_cache, relu_cache = cache
  dr = relu_backward(dout, relu_cache)
  dx, dgamma, dbeta = batchnorm_backward_alt(dr, bn_cache)
  return dx, dgamma, dbeta

def aff_bn_relu_forward(x, w, gamma, beta, bn_param):
  a, aff_cache = affine_forward_nob(x, w)
  out, out_cache = bn_relu_forward(a, gamma, beta, bn_param)
  cache = (aff_cache, out_cache)
  return out, cache

def aff_bn_relu_backward(dout, cache):
  aff_cache, out_cahce = cache
  dx, dgamma, dbeta = bn_relu_backward(dout, out_cahce)
  dx, dw = affine_backward_nob(dx, aff_cache)
  return dx, dw, dgamma, dbeta

def conv_bn_relu_pool_forward(x, w, gamma, beta, conv_param, bn_param, pool_param):
  c, cache1 = conv_bn_relu_forward(x, w, gamma, beta, conv_param, bn_param)
  out, pool_cache = max_pool_forward_fast(c, pool_param)
  cache = (cache1, pool_cache)
  return out, cache

def conv_bn_relu_pool_backward(dout, cache):
  cache1, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  dx, dw, dgamma, dbeta = conv_bn_relu_backward(ds, cache1)
  return dx, dw, dgamma, dbeta