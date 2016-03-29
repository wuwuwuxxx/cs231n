# random search for hyperparameter optimization
import random
from cs231n.classifiers.convnet import FiveLayerConvNet
import numpy as np
from cs231n.solver import Solver
from cs231n.data_utils import get_CIFAR10_data
import matplotlib.pyplot as plt
from cs231n.vis_utils import visualize_grid
from mpl_toolkits.mplot3d import Axes3D


# predict
def predict(m, xx, yy):
    scores = m.loss(xx)
    y_pred = np.argmax(scores, axis=1)
    acc = np.mean(y_pred == yy)
    return acc


def predict_train(m, xx, yy):
    n = xx.shape[0]
    step = 100
    correct = 0
    for i in range(0, n, step):
        scores = m.loss(xx[i:i+step])
        y_pred = np.argmax(scores, axis=1)
        correct += np.sum(y_pred == yy[i:i+step])
    return correct/float(n)

data = get_CIFAR10_data()

# search numbers
N = 20
history = {}
best_acc = -1

best_model = FiveLayerConvNet()

for i in range(N):
    lr = 10 ** random.uniform(-6, 1)
    reg = 10 ** random.uniform(-6, 1)
    print 'lr:', lr, 'reg:', reg
    model = FiveLayerConvNet(reg=reg)

    solver = Solver(model, data,
                    num_epochs=1, batch_size=64,
                    update_rule='adam',
                    optim_config={
                      'learning_rate': lr,
                    },
                    verbose=True, print_every=100)
    solver.train()
    train_acc = predict_train(model, data['X_train'], data['y_train'])
    val_acc = predict(model, data['X_val'], data['y_val'])
    print ' train_acc:', train_acc, ' val_acc:', val_acc
    history[(lr, reg)] = (train_acc, val_acc)
    if best_acc < val_acc:
        best_model = model
        best_acc = val_acc


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for key, value in history.iteritems():
    x, y = key
    z = value[1]
    ax.scatter(x, y, z)
plt.show()


with open('his.txt', 'w') as f:
    for k, v in history.iteritems():
        lr, reg = k
        train_acc, val_acc = v
        f.write('lr: '+str(lr)+' reg: '+str(reg)+' train_acc: '+str(train_acc)+' val_acc: '+str(val_acc) + '\n')


# visualize the weights


grid = visualize_grid(best_model.params['W1'].transpose(0, 2, 3, 1))
plt.imshow(grid.astype('uint8'))
plt.axis('off')
plt.gcf().set_size_inches(5, 5)
plt.show()
