import matplotlib.pyplot as plt

a = [1, 2, 3, 4]
b = [1, 2, 4, 8]

n = len(a)
val_acc, = plt.plot(a)
train_acc, = plt.plot(b)
plt.legend([val_acc, train_acc], ['val acc', 'train acc'])
plt.show()

