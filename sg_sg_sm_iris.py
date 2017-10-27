import numpy as np, sklearn.datasets as skld
X,y=skld.load_iris(return_X_y=True)
yoh=np.eye(len(np.unique(y)),dtype=int)[y]
np.random.seed(1)
W0,W1,W2=np.random.randn(X.shape[1],64),np.random.randn(64,64),np.random.randn(64,len(np.unique(y)))
for i in range(1000):
    a1=1./(1+np.exp(-X.dot(W0)))
    a2=1./(1+np.exp(-a1.dot(W1)))
    a3=a2.dot(W2)
    a3=np.exp(a3-np.max(a3,1,keepdims=True))
    a3/=np.sum(a3,1,keepdims=True)
    loss=-np.sum(np.log(a3[yoh==1]))/X.shape[0]
    d3=(a3-yoh)/X.shape[0]
    d2=d3.dot(W2.T)*a2*(1-a2)
    d1=d2.dot(W1.T)*a1*(1-a1)
    W2-=a2.T.dot(d3)
    W1-=a1.T.dot(d2)
    W0-=X.T.dot(d1)
#print("LOSS({})\nPREDICT\n{}".format(loss,a3))
print("LOSS({})".format(loss))
print("0P ({}) 1P ({}) 2P({})".format(np.average(a3[0:50,0]),np.average(a3[50:100,1]),np.average(a3[100:150,2])))
