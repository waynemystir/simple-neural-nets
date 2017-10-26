import numpy as np, sklearn.datasets as skld
X,y = skld.load_iris(return_X_y=True)
X = X.T
y_onehot = np.eye(3,dtype=int)[y].T
np.random.seed(1)
W0,W1 = np.random.randn(128, 4),np.random.randn(3, 128)
for i in range(1000):
    a1 = 1./(1+np.exp(-W0.dot(X)))
    a2 = 1./(1+np.exp(-W1.dot(a1)))
    loss = np.sum(np.power(y_onehot-a2,2))/2/X.shape[1]
    l2_delta = (a2-y_onehot) * a2 * (1 - a2)/X.shape[1]
    l1_delta = W1.T.dot(l2_delta) * a1 * (1 - a1)
    W1 -= l2_delta.dot(a1.T)
    W0 -= l1_delta.dot(X.T)
print("Final loss ({}) prediction \n{}".format(loss,a2))
