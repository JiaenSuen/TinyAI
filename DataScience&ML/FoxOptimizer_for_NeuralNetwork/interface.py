'''
The Catalog for this tiny project
* Data Preprocessing
* Initialization
* Forward Propagation
* Activation Functions (ReLU and Softmax)
* Backpropagation
* Parameter Updates (Gradient Descent)

## New
* Adam
* FOA
'''

from modules.load_mnist import load_mnist_inCSV
from modules.neuralnet  import NeuralNet
from modules.FoxOptimizer import FoxOptimizer
from modules.base import *
import matplotlib.pyplot as plt



TRAIN_DATASET_PATH = "mnist/mnist_train.csv"
TEST_DATASET_PATH  = "mnist/mnist_test.csv"
X_train, Y_train = load_mnist_inCSV(TRAIN_DATASET_PATH)
X_test , Y_test  = load_mnist_inCSV(TEST_DATASET_PATH)









def test_prediction(index, model, dataset_X, dataset_Y):

    current_image = dataset_X[:, index, None]

    prediction = model.predict(current_image)
    label = dataset_Y[index]

    print(f"Index : {index}")
    print("Prediction:", prediction)
    print("Label:", label)
    print()

    current_image = current_image.reshape((28, 28)) * 255

    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()



def objective(params):

    lr = params[0]
    hidden = int(params[1])

    subset = 5000
    X_small = X_train[:, :subset]
    Y_small = Y_train[:subset]

    model = NeuralNet([784, hidden, 10])

    model.train(X_small, Y_small,
                iterations=50,
                lr=lr,
                show=False)

    preds = model.predict(X_test)

    acc = model.get_accuracy(preds, Y_test)

 
    print(f"{GREEN} trial : {lr:.20f} , {hidden:3d} | acc: {acc}{RESET}")
    return acc



if __name__ == "__main__":

    ExpNum= input("Chose experiement code (1) : Training NN as you set. (2) Hyperparameter Search by FOA. :  ")
    if ExpNum == "1":
        
        hidden_unit = eval(input("Set Hidden Layer Units : "))
        lr = eval(input("Set Learning Rate : "))
        model = NeuralNet([784,hidden_unit,10]) # Original : 10


        model.train(X_train, Y_train, iterations=100, lr=lr ) # Original : 0.1

        predictions = model.predict(X_test)

        print("Test accuracy:", model.get_accuracy(predictions, Y_test,out_path="confusion_matrix.png"))

        test_prediction(0, model, X_test, Y_test)
        test_prediction(1, model, X_test, Y_test)
        test_prediction(2, model, X_test, Y_test)
        test_prediction(3, model, X_test, Y_test)

    elif ExpNum == "2":
        

        bounds = [
            (0.0001,0.01),   # lr
            (64,512)         # hidden
        ]

        foa = FoxOptimizer(
            pop_size=8,
            iterations=5 #20
        )


        best = foa.optimize(objective, bounds)
        best_lr = best[0]
        best_hidden = int(best[1])
        print(RED + f"Best hyperparameters: lr={best_lr:.6f}, hidden={best_hidden}" + RESET)


 # Original : Test accuracy: 0.9113 , train : 90.91
 # Original + Adam : Test accuracy: 0.9263 , train : 94.23
 # FOA Search :  