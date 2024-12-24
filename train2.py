# intended to implement digit classification

import math
import numpy as np
from mlxtend.data import loadlocal_mnist

# importing dataset
X, y = loadlocal_mnist(images_path='./data/train-images.idx3-ubyte', labels_path='./data/train-labels.idx1-ubyte')

# spliting dataset
num_train = 50000
num_test = 10000
X_train = X[:num_train, :] / 255
y_train = np.zeros((num_train, 10))
y_train[np.arange(0, num_train), y[:num_train]] = 1  

X_test = X[num_train:, :] / 255
y_test = np.zeros((num_test, 10))
y_test[np.arange(0, num_test), y[y.size - num_test:]] = 1

print("Training set shape : ", X_train.shape)
print("Test set shape : ", X_test.shape)

batch_size = 16

weights = np.random.random(784) # 784 pixels (28x28)
weights -= .5

best_weights_yet = []
final_weights = []

def sigmoid(x):
    return (1/(1 + math.e ** (-x)))

for N in range(10): # for each digit N
    alpha = .02
    best_off_by_yet = 100
    cost = []
    xval = 0
    offset = N % batch_size
    for _ in range(10): # susceptible to overfitting
        for batch in range(num_train // batch_size - 1):
            X = X_train[batch * batch_size + offset:(batch + 1) * batch_size + offset]
            y_batch = y_train[batch * batch_size + offset:(batch + 1) * batch_size + offset]
            labels = [y_batch[i][N] for i in range(batch_size)]

            thing = np.dot(X, weights) # hypothesized values pre sigmoid
            sigmoided = sigmoid(thing)
            off_by = sigmoided - [a[N] for a in y_batch]
            if sum(off_by ** 2) < best_off_by_yet:
                best_weights_yet = weights
            best_off_by_yet = min(sum(off_by ** 2), best_off_by_yet)
            
            weights -= alpha / batch_size * np.matmul(np.transpose(X), (off_by))
            cost.append(sum(off_by ** 2))
            xval += 1
        alpha *= .98

    final_weights.append(best_weights_yet)
    weights = np.random.random(784) # 784 pixels (28x28)
    weights -= .5
    print(str((N + 1) * 10) + "%" + " done training")


how_many_right = 0
mismatches = []

for i in range(num_test):
    test = X_test[i]
    answer = np.where(y_test[i] == 1)[0]

    predicted = [[] for _ in range(10)]
    for i in range(10):
        predicted[i] = sigmoid(np.dot(test, final_weights[i]))
    # print(predicted)
    # print(answer)
    guess = predicted.index(max(predicted))
    if guess == answer:
        how_many_right += 1
    else:
        mismatches.append((guess, answer))


print(how_many_right)
np.savetxt("weights.csv", final_weights, delimiter=",")