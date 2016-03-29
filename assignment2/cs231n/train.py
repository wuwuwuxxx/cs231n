from cs231n.vis_utils import visualize_grid
import matplotlib.pyplot as plt
from cs231n.solver import Solver
from cs231n.data_utils import get_CIFAR10_data
from cs231n.classifiers.convnet import FiveLayerConvNet
import numpy as np


# predict
def predict(m, xx, yy):
    scores = m.loss(xx, mode='test')
    y_pred = np.argmax(scores, axis=1)
    acc = np.mean(y_pred == yy)
    return acc


def predict_train(m, xx, yy):
    n = xx.shape[0]
    step = 100
    correct = 0
    for i in range(0, n, step):
        scores = m.loss(xx[i:i+step], mode='test')
        y_pred = np.argmax(scores, axis=1)
        correct += np.sum(y_pred == yy[i:i+step])
    return correct/float(n)

data = get_CIFAR10_data()

model = FiveLayerConvNet(reg=0.0005)

solver = Solver(model, data,
                num_epochs=75, batch_size=64,
                update_rule='rmsprop',
                optim_config={
                  'learning_rate': 1e-2,
                },
                lr_decay=0.1,
                verbose=True, print_every=100)
solver.train()

train_acc = predict_train(model, data['X_train'], data['y_train'])
val_acc = predict(model, data['X_val'], data['y_val'])
test_acc = predict(model, data['X_test'], data['y_test'])
print ' train_acc:', train_acc, ' val_acc:', val_acc, ' test_acc', test_acc

plt.subplot(2, 1, 1)
p_train, = plt.plot(solver.train_acc_history)
p_val, = plt.plot(solver.val_acc_history)
plt.legend([p_train, p_val], ['training accuracy', 'validation accuracy'])

plt.subplot(2, 1, 2)
grid = visualize_grid(model.params['W1'].transpose(0, 2, 3, 1))
plt.imshow(grid.astype('uint8'))
plt.axis('off')
plt.gcf().set_size_inches(5, 5)
plt.show()
