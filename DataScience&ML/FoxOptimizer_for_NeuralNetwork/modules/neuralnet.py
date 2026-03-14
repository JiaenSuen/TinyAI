import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix



class ActivationFunction:

    @staticmethod
    def ReLU(Z):
        return np.maximum(0,Z)
    @staticmethod
    def Deriv_ReLU(Z):
        return (Z > 0).astype(float)
    

    @staticmethod
    def Softmax(Z): # fix : avoid overflow
        Z = Z - np.max(Z, axis=0, keepdims=True)
        expZ = np.exp(Z)
        return expZ / np.sum(expZ, axis=0, keepdims=True) # sum for each column across all of the rows in that column and divide each element by that -> and get rthe pobability we want.

    



class NeuralNet():
 
    def __init__(self, layers):
         # Parameters Initialization by " He initialization ", reference : https://www.kaggle.com/discussions/general/299381
 
        self.layers = layers
        self.L = len(layers) - 1

        self.W = []
        self.b = []

 
        for l in range(self.L):
            in_dim  = layers[l]
            out_dim = layers[l+1]

            # He initialization
            W = np.random.randn(out_dim, in_dim) * np.sqrt(2 / in_dim)
            b = np.zeros((out_dim, 1))

            self.W.append(W)
            self.b.append(b)

 
        self.mW = [np.zeros_like(W) for W in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vW = [np.zeros_like(W) for W in self.W]
        self.vb = [np.zeros_like(b) for b in self.b]

        self.t = 0
 
    # Utility
    def one_hot(self, Y):
        one_hot_Y = np.zeros((10, Y.size))
        one_hot_Y[Y, np.arange(Y.size)] = 1
        return one_hot_Y

    # Interface
    def predict(self, X):
        Z_cache, A_cache = self.forward(X)
        preds = np.argmax(A_cache[-1],0)
        return preds


    def get_accuracy(self, predictions, Y, out_path=None):

        acc = np.sum(predictions == Y) / Y.size

        if out_path is not None:

            cm = confusion_matrix(Y, predictions)

            plt.figure(figsize=(6,6))
            sns.heatmap(cm,
                        annot=True,
                        fmt="d",
                        cmap="Blues")

            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix")

            plt.savefig(out_path)
            plt.close()

        return acc





    def forward(self, X):

        A = X
        Z_cache = []
        A_cache = [X]

        for l in range(self.L):

            Z = self.W[l] @ A + self.b[l]

            if l == self.L-1:
                A = ActivationFunction.Softmax(Z)
            else:
                A = ActivationFunction.ReLU(Z)

            Z_cache.append(Z)
            A_cache.append(A)

        return Z_cache, A_cache





    def backward(self, Z_cache, A_cache, X, Y):

        m = Y.size
        one_hot_Y = self.one_hot(Y)

        dW = [None]*self.L
        db = [None]*self.L

        dZ = A_cache[-1] - one_hot_Y

        for l in reversed(range(self.L)):

            A_prev = A_cache[l]

            dW[l] = 1/m * dZ @ A_prev.T
            db[l] = 1/m * np.sum(dZ, axis=1, keepdims=True)

            if l > 0:
                dA_prev = self.W[l].T @ dZ
                dZ = dA_prev * ActivationFunction.Deriv_ReLU(Z_cache[l-1])

        return dW, db

    
    def update(self, dW, db, lr):

        for l in range(self.L):

            self.W[l] -= lr * dW[l]
            self.b[l] -= lr * db[l]

    def update_adam(self, dW, db, lr,
                beta1=0.9,
                beta2=0.999,
                eps=1e-8):

        self.t += 1

        for l in range(self.L):

            self.mW[l] = beta1*self.mW[l] + (1-beta1)*dW[l]
            self.mb[l] = beta1*self.mb[l] + (1-beta1)*db[l]

            self.vW[l] = beta2*self.vW[l] + (1-beta2)*(dW[l]**2)
            self.vb[l] = beta2*self.vb[l] + (1-beta2)*(db[l]**2)

            mW_hat = self.mW[l] / (1-beta1**self.t)
            mb_hat = self.mb[l] / (1-beta1**self.t)

            vW_hat = self.vW[l] / (1-beta2**self.t)
            vb_hat = self.vb[l] / (1-beta2**self.t)

            self.W[l] -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            self.b[l] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)





    def train(self, X, Y, iterations, lr, log_file="train_acc.txt",show=True):

        with open(log_file, "w") as f:

             for i in range(1,iterations+1):

                Z_cache, A_cache = self.forward(X)

                dW, db = self.backward(Z_cache, A_cache, X, Y)

                self.update_adam(dW, db, lr)


           

                if i % 10 == 0 and show:
                    preds = np.argmax(A_cache[-1],0)
                    acc = self.get_accuracy(preds, Y)
                    f.write(f"{i} {acc}\n")
                    print("iteration: %3d ; accuracy: %.4f" % (i, acc))