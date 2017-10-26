import numpy as np
X,y = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]]),np.array([0,1,1,0])
y_onehot = np.eye(2,dtype=int)[y]
np.random.seed(1)
W0,W1 = np.random.randn(3, 4),np.random.randn(4, 2)
for i in range(1000):
    a1 = 1./(1+np.exp(-X.dot(W0)))
    a2 = a1.dot(W1)
    a2 = np.exp(a2 - np.max(a2,1,keepdims=True))
    a2 /= np.sum(a2,1,keepdims=True)
    loss = -np.sum(np.log(a2[y_onehot==1]))/X.shape[0]
    l2_delta = (a2-y_onehot)/X.shape[0]
    l1_delta = l2_delta.dot(W1.T) * a1 * (1 - a1)
    W1 -= a1.T.dot(l2_delta)
    W0 -= X.T.dot(l1_delta)
print("Final loss ({}) prediction \n{}".format(loss,a2))
