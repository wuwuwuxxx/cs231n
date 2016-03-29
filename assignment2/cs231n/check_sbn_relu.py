# Train a really good model on CIFAR-10
from cs231n.layer_utils import sbn_relu_forward, sbn_relu_backward
import numpy as np
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

x = np.random.randn(2, 3, 8, 8)
dout = np.random.randn(2, 3, 8, 8)
bn_param = {'mode': 'train'}
gamma = np.random.randn(3)
beta = np.random.randn(3)

out, cache = sbn_relu_forward(x, gamma, beta, bn_param)
dx, dgamma, dbeta = sbn_relu_backward(dout, cache)

dx_num = eval_numerical_gradient_array(lambda x: sbn_relu_forward(x, gamma, beta,
                                                                     bn_param)[0], x, dout)
dgamma_num = eval_numerical_gradient_array(lambda gamma: sbn_relu_forward(x, gamma, beta,
                                                                     bn_param)[0], gamma, dout)
dbeta_num = eval_numerical_gradient_array(lambda beta: sbn_relu_forward(x, gamma, beta,
                                                                     bn_param)[0], beta, dout)
print 'Testing conv_relu:'
print 'dx error: ', rel_error(dx_num, dx)
print 'dgamma error: ', rel_error(dgamma_num, dgamma)
print 'dbeta error: ', rel_error(dbeta_num, dbeta)