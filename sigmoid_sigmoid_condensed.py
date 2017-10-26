import numpy as np
X,y = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]]),np.array([0,1,1,0])
y_onehot = np.eye(2,dtype=int)[y]
np.random.seed(1)
W0,W1 = np.random.randn(3, 8),np.random.randn(8, 2)
for i in range(2000):
    a1 = 1./(1+np.exp(-X.dot(W0)))
    a2 = 1./(1+np.exp(-a1.dot(W1)))
    loss = np.sum(np.power(y_onehot-a2,2))/2/X.shape[0]
    l2_delta = (a2-y_onehot) * a2 * (1 - a2)/X.shape[0]
    l1_delta = l2_delta.dot(W1.T) * a1 * (1 - a1)
    W1 -= a1.T.dot(l2_delta)
    W0 -= X.T.dot(l1_delta)
print("Final loss ({}) prediction \n{}".format(loss,a2))
