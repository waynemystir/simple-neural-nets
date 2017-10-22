import argparse
import numpy as np
from simple_utils import sigmoid, sigmoid_grad, quadratic_loss, quadratic_grad
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Simple two layer neural net script with sigmoid activation')
parser.add_argument('--plot', action='store_true',
                            help='plot the loss and weights')
args = parser.parse_args()

X = np.array([
    [1,1,1,1],
    [0,1,0,1],
    [0,0,1,1]
    ])

y = np.array([[0,0,1,1]])

# seed random number to make the calculation
# deterministic (easier to debug, etc)
np.random.seed(1)

W = np.random.randn(1, 3)
Ws = W.copy()

iz = []
losses = []

for i in range(1000):

    # forward propagate
    a1 = sigmoid(W.dot(X))
    loss = quadratic_loss(y, a1)

    # backpropagation

    # how much we missed times the slope
    # of the sigmoid at the values in a1
    delta = quadratic_grad(y, a1) * sigmoid_grad(a1)

    # loss due to weights
    nabla_w = delta.dot(X.T)

    # and update the weights
    W -= nabla_w

    iz.append(i)
    losses.append(loss)
    if i != 0: Ws = np.concatenate((Ws, W.copy()), axis = 0)

print("Final prediction ({})".format(a1))
print("Final W ({})".format(W))

if args.plot:
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    lss, = ax1.plot(iz, losses, 'r-', label = 'loss')
    Ws0, = ax2.plot(iz, Ws[:,0], label='W0')
    Ws1, = ax2.plot(iz, Ws[:,1], label='W1')
    Ws2, = ax2.plot(iz, Ws[:,2], label='W2')
    plt.legend(handles=[lss, Ws0, Ws1, Ws2])
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss', color='r')
    ax2.set_ylabel('Weights')
    ax2
    plt.show()
