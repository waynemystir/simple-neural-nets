import numpy as np
import random

def sigmoid(x):
    return 1./(1 + np.exp(-x))

def sigmoid_grad(f):
    return f * (1 - f)

def quadratic_loss(y, a):
    return 1/2 * np.sum(np.power(y - a, 2))

def quadratic_grad(y, a):
    return a - y

def softmax(x):
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        x = np.exp(x - np.max(x,1,keepdims=True))
        x /= np.sum(x,1,keepdims=True)
    else:
        #Vector
        x = np.exp(x - np.max(x))
        x /= np.sum(g)

    assert x.shape == orig_shape
    return x

def cross_entropy_loss(y, yhat):
#    return -np.sum(np.nan_to_num(np.log(yhat[range(4), y.reshape(4)].clip(min=1e-4))))/y.shape[0]
    return -np.sum(np.nan_to_num(np.log(yhat[y==1].clip(min=1e-4))))/y.shape[0]

def cross_entropy_grad(y, yhat):
    return (yhat - y)/y.shape[0]

def gradcheck_naive(f, x):
    """ Gradient check for a function f.

    Arguments:
    f -- a function that takes a single argument and outputs the
         cost and its gradients
    x -- the point (numpy array) to check the gradient at
    """

    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4        # Do not change this!

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        # Try modifying x[ix] with h defined above to compute
        # numerical gradients. Make sure you call random.setstate(rndstate)
        # before calling f(x) each time. This will make it possible
        # to test cost functions with built in randomness later.

        x[ix] += h
        random.setstate(rndstate)
        f1 = f(x)[0]

        x[ix] -= 2*h
        random.setstate(rndstate)
        f2 = f(x)[0]

        x[ix] += h
        numgrad = (f1 - f2)/(2 * h)

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print("Gradient check failed.")
            print("First gradient error found at index %s" % str(ix))
            print("Your gradient: %f \t Numerical gradient: %f" % (
                grad[ix], numgrad))
            return

        it.iternext() # Step to next dimension

    print("Gradient check passed!")


def gradcheck_sanity_check():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print("Running sanity checks...")
    gradcheck_naive(quad, np.array(123.456))      # scalar test
    gradcheck_naive(quad, np.random.randn(3,))    # 1-D test
    gradcheck_naive(quad, np.random.randn(4,5))   # 2-D test
    print("")

if __name__ == "__main__":
    gradcheck_sanity_check()
