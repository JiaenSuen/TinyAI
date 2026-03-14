import numpy as np

from .neuralnet import (
    NeuralNet,
    ActivationFunction,
)


# -- Announcement --
# Object-oriented version iterations happen too quickly; 
# let's just treat this as ruins.
# Version update information: 
# Replacing redundant parameters and function passing with object-oriented programming,
# and exploring the known boundaries of the universe.


'''

def back_propagation(Z1, A1, Z2, A2, W2 ,X, Y):
    m = Y.size 
    one_hot_Y = one_hot(Y)

    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * ActivationFunction.Deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2


def update_params(W1,b1, W2,b2 , dW1,db1,dW2,db2, alpha ):
    W1 = W1 - alpha*dW1
    b1 = b1 - alpha*db1
    W2 = W2 - alpha*dW2
    b2 = b2 - alpha*db2
    return W1,b1,W2,b2





def GradientDescent(X,Y, iterations , alpha):
    W1,b1,W2,b2 =  init_params()
    for i in range(iterations):
        z1, a1, z2, a2 = forward_propagation(W1,b1,W2,b2, X)
        dW1, db1, dW2, db2 = back_propagation(z1, a1, z2, a2, W2, X, Y)
        W1,b1,W2,b2 = update_params(W1,b1,W2,b2, dW1, db1, dW2, db2, alpha)
        if i%10==0:
            print("iterations: ",i)
            print("accuracy: ",get_accuracy(get_predictions(a2),Y))
    return W1,b1,W2,b2
'''