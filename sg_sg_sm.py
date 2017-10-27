import numpy as np
X,y=np.array([[0,0,1],[1,0,1],[0,1,1],[1,1,1],]),np.array([0,1,1,0])
yoh=np.eye(2,dtype=int)[y]
np.random.seed(1)
W0,W1,W2=np.random.randn(3,64),np.random.randn(64,64),np.random.randn(64,2)
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
print("LOSS({})\nPREDICT\n{}".format(loss,a3))
