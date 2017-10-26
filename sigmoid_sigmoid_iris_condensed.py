import numpy as np, sklearn.datasets as skld
X,y = skld.load_iris(return_X_y=True)
y_onehot = np.eye(3,dtype=int)[y]
np.random.seed(1)
W0,W1 = np.random.randn(4, 64),np.random.randn(64, 3)
for i in range(1000):
    a1 = 1./(1+np.exp(-X.dot(W0)))
    a2 = 1./(1+np.exp(-a1.dot(W1)))
    loss = np.sum(np.power(y_onehot-a2,2))/X.shape[0]
    l2_delta = (a2-y_onehot) * a2 * (1 - a2)/X.shape[0]
    l1_delta = l2_delta.dot(W1.T) * a1 * (1 - a1)
    W1 -= a1.T.dot(l2_delta)
    W0 -= X.T.dot(l1_delta)
print("Final loss ({}) prediction \n{}".format(loss,a2))
