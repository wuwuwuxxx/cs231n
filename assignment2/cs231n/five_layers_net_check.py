import numpy as np
from cs231n.classifiers.convnet import FiveLayerConvNet
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

model = FiveLayerConvNet()


N = 50
X = np.random.randn(N, 3, 32, 32)
y = np.random.randint(10, size=N)

loss, _ = model.loss(X, y)
print 'Initial loss (no regularization): ', loss

model.reg = 0.0005
loss, _ = model.loss(X, y)
print 'Initial loss (with regularization): ', loss

num_inputs = 2
input_dim = (3, 8, 8)
reg = 0.0
num_classes = 10
X = np.random.randn(num_inputs, *input_dim)
y = np.random.randint(num_classes, size=num_inputs)

model = FiveLayerConvNet(input_dim=input_dim, reg=reg)
loss, grads = model.loss(X, y)
for param_name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    param_grad_num = eval_numerical_gradient(f, model.params[param_name], verbose=False, h=1e-6)
    e = rel_error(param_grad_num, grads[param_name])
    print '%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name]))