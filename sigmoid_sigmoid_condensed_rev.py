import numpy as np
X,y = np.array([[0,0,1,1],[0,1,0,1],[1,1,1,1]]),np.array([0,1,1,0])
y_onehot = np.eye(2,dtype=int)[y].T
np.random.seed(1)
W0,W1 = np.random.randn(8, 3),np.random.randn(2, 8)
for i in range(2000):
    a1 = 1./(1+np.exp(-W0.dot(X)))
    a2 = 1./(1+np.exp(-W1.dot(a1)))
    loss = np.sum(np.power(y_onehot-a2,2))/2/X.shape[1]
    l2_delta = (a2-y_onehot) * a2 * (1 - a2)/X.shape[1]
    l1_delta = W1.T.dot(l2_delta) * a1 * (1 - a1)
    W1 -= l2_delta.dot(a1.T)
    W0 -= l1_delta.dot(X.T)
print("Final loss ({}) prediction \n{}".format(loss,a2))
