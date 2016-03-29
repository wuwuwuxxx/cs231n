import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class FiveLayerConvNet(object):
  """
  A five-layer convolutional network with the following architecture:

  conv1-BN1-relu1-conv2-BN2-relu2-pool1-conv3-BN3-relu3-pool2-affine1-BN4-relu-affine-softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_classes=10, reg=0.0,
               dtype=np.float32, mode='train'):
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
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    self.bn_param1 = {'mode': mode}
    self.bn_param2 = {'mode': mode}
    self.bn_param3 = {'mode': mode}
    self.bn_param4 = {'mode': mode}
    self.bn_param5 = {'mode': mode}
    self.bn_param6 = {'mode': mode}

    # network architecture
    filter_size1 = 3
    num_filters1 = 32
    weight_scale1 = np.sqrt(2.0/(input_dim[0] * filter_size1 ** 2))
    self.conv1_param = {'stride': 1, 'pad': (filter_size1-1)/2}

    filter_size2 = 3
    num_filters2 = 32
    weight_scale2 = np.sqrt(2.0/(num_filters1 * filter_size2 ** 2))
    self.conv2_param = {'stride': 1, 'pad': (filter_size2-1)/2}

    filter_size3 = 3
    num_filters3 = 64
    weight_scale3 = np.sqrt(2.0/(num_filters3 * filter_size3 ** 2))
    self.conv3_param = {'stride': 1, 'pad': (filter_size3-1)/2}

    filter_size4 = 3
    num_filters4 = 64
    weight_scale4 = np.sqrt(2.0/(num_filters4 * filter_size4 ** 2))
    self.conv4_param = {'stride': 1, 'pad': (filter_size4-1)/2}

    filter_size5 = 3
    num_filters5 = 128
    weight_scale5 = np.sqrt(2.0/(num_filters5 * filter_size5 ** 2))
    self.conv5_param = {'stride': 1, 'pad': (filter_size5-1)/2}

    filter_size6 = 3
    num_filters6 = 128
    weight_scale6 = np.sqrt(2.0/(num_filters6 * filter_size6 ** 2))
    self.conv6_param = {'stride': 1, 'pad': (filter_size6-1)/2}

    weight_scale7 = np.sqrt(2.0/(num_filters6*input_dim[1]**2/4))

    # initialize
    self.params['W1'] = np.random.normal(scale=weight_scale1, size=(num_filters1, input_dim[0], filter_size1, filter_size1))
    self.params['gamma1'] = np.random.rand(num_filters1)
    self.params['beta1'] = np.random.rand(num_filters1)

    self.params['W2'] = np.random.normal(scale=weight_scale2, size=(num_filters2, num_filters1, filter_size2, filter_size2))
    self.params['gamma2'] = np.random.rand(num_filters2)
    self.params['beta2'] = np.random.rand(num_filters2)

    self.params['W3'] = np.random.normal(scale=weight_scale3, size=(num_filters3, num_filters2, filter_size3, filter_size3))
    self.params['gamma3'] = np.random.rand(num_filters3)
    self.params['beta3'] = np.random.rand(num_filters3)

    self.params['W4'] = np.random.normal(scale=weight_scale4, size=(num_filters4, num_filters3, filter_size4, filter_size4))
    self.params['gamma4'] = np.random.rand(num_filters4)
    self.params['beta4'] = np.random.rand(num_filters4)

    self.params['W5'] = np.random.normal(scale=weight_scale5, size=(num_filters5, num_filters4, filter_size5, filter_size5))
    self.params['gamma5'] = np.random.rand(num_filters5)
    self.params['beta5'] = np.random.rand(num_filters5)

    self.params['W6'] = np.random.normal(scale=weight_scale6, size=(num_filters6, num_filters5, filter_size6, filter_size6))
    self.params['gamma6'] = np.random.rand(num_filters6)
    self.params['beta6'] = np.random.rand(num_filters6)

    self.params['W7'] = np.random.normal(scale=weight_scale7, size=(num_filters6*(input_dim[1]/8)**2, num_classes))
    self.params['b7'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None, mode='train'):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    self.bn_param1['mode'] = mode
    self.bn_param2['mode'] = mode
    self.bn_param3['mode'] = mode
    self.bn_param4['mode'] = mode
    self.bn_param5['mode'] = mode
    self.bn_param6['mode'] = mode

    W1 = self.params['W1']
    W2 = self.params['W2']
    W3 = self.params['W3']
    W4 = self.params['W4']
    W5 = self.params['W5']
    W6 = self.params['W6']
    W7 = self.params['W7']
    b7 = self.params['b7']
    gamma1, beta1 = self.params['gamma1'], self.params['beta1']
    gamma2, beta2 = self.params['gamma2'], self.params['beta2']
    gamma3, beta3 = self.params['gamma3'], self.params['beta3']
    gamma4, beta4 = self.params['gamma4'], self.params['beta4']
    gamma5, beta5 = self.params['gamma5'], self.params['beta5']
    gamma6, beta6 = self.params['gamma6'], self.params['beta6']

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    # dropout_param = {'p': 0.5, 'mode': mode}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the seven-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    layer1_out, layer1_cache = conv_bn_relu_forward(X, W1, gamma1, beta1, self.conv1_param, self.bn_param1)
    # drop1_out, drop1_cache = dropout_forward(layer1_out, dropout_param)
    layer2_out, layer2_cache = conv_bn_relu_pool_forward(layer1_out, W2, gamma2, beta2, self.conv2_param, self.bn_param2, pool_param)

    layer3_out, layer3_cache = conv_bn_relu_forward(layer2_out, W3, gamma3, beta3, self.conv3_param, self.bn_param3)
    # drop2_out, drop2_cache = dropout_forward(layer3_out, dropout_param)
    layer4_out, layer4_cache = conv_bn_relu_pool_forward(layer3_out, W4, gamma4, beta4, self.conv4_param, self.bn_param4, pool_param)

    layer5_out, layer5_cache = conv_bn_relu_forward(layer4_out, W5, gamma5, beta5, self.conv5_param, self.bn_param5)
    # drop3_out, drop3_cache = dropout_forward(layer5_out, dropout_param)
    layer6_out, layer6_cache = conv_bn_relu_pool_forward(layer5_out, W6, gamma6, beta6, self.conv6_param, self.bn_param6, pool_param)
    scores, scores_cache = affine_forward(layer6_out, W7, b7)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dx = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.linalg.norm(W1)**2 + np.linalg.norm(W2)**2 + np.linalg.norm(W3)**2 + np.linalg.norm(W4)**2 + np.linalg.norm(W5)**2 + np.linalg.norm(W6)**2 + np.linalg.norm(W7)**2)
    dx, grads['W7'], grads['b7'] = affine_backward(dx, scores_cache)
    dx, grads['W6'], grads['gamma6'], grads['beta6'] = conv_bn_relu_pool_backward(dx, layer6_cache)
    # dx = dropout_backward(dx, drop3_cache)
    dx, grads['W5'], grads['gamma5'], grads['beta5'] = conv_bn_relu_backward(dx, layer5_cache)
    dx, grads['W4'], grads['gamma4'], grads['beta4'] = conv_bn_relu_pool_backward(dx, layer4_cache)
    # dx = dropout_backward(dx, drop2_cache)
    dx, grads['W3'], grads['gamma3'], grads['beta3'] = conv_bn_relu_backward(dx, layer3_cache)
    dx, grads['W2'], grads['gamma2'], grads['beta2'] = conv_bn_relu_pool_backward(dx, layer2_cache)
    # dx = dropout_backward(dx, drop1_cache)
    _, grads['W1'], grads['gamma1'], grads['beta1'] = conv_bn_relu_backward(dx, layer1_cache)

    grads['W7'] += self.reg * W7
    grads['W6'] += self.reg * W6
    grads['W5'] += self.reg * W5
    grads['W4'] += self.reg * W4
    grads['W3'] += self.reg * W3
    grads['W2'] += self.reg * W2
    grads['W1'] += self.reg * W1
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


pass
