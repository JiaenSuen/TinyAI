import matplotlib.pyplot as plt
def plot_training_curve(file_path):

    iterations = []
    accuracies = []

    with open(file_path, "r") as f:
        for line in f:
            i, acc = line.split()
            iterations.append(int(i))
            accuracies.append(float(acc))

    plt.plot(iterations, accuracies)

    plt.xlabel("Iteration")
    plt.ylabel("Train Accuracy")
    plt.title("Training Accuracy Curve")

    plt.grid(True)
    plt.savefig("Record_Training_Curve.png")
    plt.show()



if __name__ == "__main__":

    plot_training_curve("train_acc.txt")