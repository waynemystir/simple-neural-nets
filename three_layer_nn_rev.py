import argparse
import numpy as np
from simple_utils import sigmoid, sigmoid_grad, quadratic_loss, quadratic_grad
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Simple three layer neural net script with sigmoid activations')
parser.add_argument('--plot', action='store_true',
                            help='plot the loss and weights')
args = parser.parse_args()

X = np.array([
    [0,0,1,1],
    [0,1,0,1],
    [1,1,1,1]
    ])

y = np.array([[0,1,1,0]])

# seed random number to make the calculation
# deterministic (easier to debug, etc)
np.random.seed(1)

epochs = 2000
W0 = np.random.randn(4, 3)
W1 = np.random.randn(1, 4)
W0s = W0.copy()
W1s = W1.copy()

iz = []
losses = []

for i in range(epochs):

    # forward propagate
    a1 = sigmoid(W0.dot(X))
    a2 = sigmoid(W1.dot(a1))
    loss = quadratic_loss(y, a2)

    # backpropagation

    # how much we missed times the slope
    # of the sigmoid at the values in a1
    l2_delta = quadratic_grad(y, a2) * sigmoid_grad(a2)

    # how much did each l1 value contribute to the l2 loss
    # (according to the weights)?
    l1_loss = W1.T.dot(l2_delta)

    # in what direction is the target a1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_loss * sigmoid_grad(a1)

    # loss due to weights
    nabla_w1 = l2_delta.dot(a1.T)
    nabla_w0 = l1_delta.dot(X.T)

    # and update the weights
    W1 -= nabla_w1
    W0 -= nabla_w0

    iz.append(i)
    losses.append(loss)
    if i != 0:
        W0s = np.concatenate((W0s, W0.copy()), axis = 1)
        W1s = np.concatenate((W1s, W1.copy()), axis = 1)

W0s = W0s.reshape(4, epochs, 3)
print("Final prediction ({})".format(a2))
print("Final W ({})".format(W0))

if args.plot:
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    Ws00, = ax2.plot(iz, W0s[0,:,0], label='W00')
    Ws10, = ax2.plot(iz, W0s[1,:,0], label='W10')
    Ws20, = ax2.plot(iz, W0s[2,:,0], label='W20')
    Ws30, = ax2.plot(iz, W0s[3,:,0], label='W30')
    Ws01, = ax2.plot(iz, W0s[0,:,1], label='W01')
    Ws11, = ax2.plot(iz, W0s[1,:,1], label='W11')
    Ws21, = ax2.plot(iz, W0s[2,:,1], label='W21')
    Ws31, = ax2.plot(iz, W0s[3,:,1], label='W31')
    Ws02, = ax2.plot(iz, W0s[0,:,2], label='W02')
    Ws12, = ax2.plot(iz, W0s[1,:,2], label='W12')
    Ws22, = ax2.plot(iz, W0s[2,:,2], label='W22')
    Ws32, = ax2.plot(iz, W0s[3,:,2], label='W32')
    lss, = ax1.plot(iz, losses, 'r-', label = 'loss', linewidth=3)
    plt.legend(handles=[lss, Ws00, Ws10, Ws20, Ws30, Ws01, Ws11, Ws21])
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss', color='r')
    ax2.set_ylabel('Weights')
    ax2
    plt.show()
