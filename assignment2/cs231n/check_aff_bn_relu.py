# Train a really good model on CIFAR-10
from cs231n.layer_utils import aff_bn_relu_forward, aff_bn_relu_backward
import numpy as np
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

x = np.random.randn(5, 6)
w = np.random.randn(6, 7)
dout = np.random.randn(5, 7)
bn_param = {'mode': 'train'}
gamma = np.random.randn(7)
beta = np.random.randn(7)

out, cache = aff_bn_relu_forward(x, w, gamma, beta, bn_param)
dx, dw, dgamma, dbeta = aff_bn_relu_backward(dout, cache)

dx_num = eval_numerical_gradient_array(lambda x: aff_bn_relu_forward(x, w, gamma, beta,
                                                                     bn_param)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: aff_bn_relu_forward(x, w, gamma, beta,
                                                                     bn_param)[0], w, dout)
dgamma_num = eval_numerical_gradient_array(lambda gamma: aff_bn_relu_forward(x, w, gamma, beta,
                                                                     bn_param)[0], gamma, dout)
dbeta_num = eval_numerical_gradient_array(lambda beta: aff_bn_relu_forward(x, w, gamma, beta,
                                                                     bn_param)[0], beta, dout)
print 'Testing conv_relu:'
print 'dx error: ', rel_error(dx_num, dx)
print 'dw error: ', rel_error(dw_num, dw)
print 'dgamma error: ', rel_error(dgamma_num, dgamma)
print 'dbeta error: ', rel_error(dbeta_num, dbeta)
