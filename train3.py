# intended to implement digit classification

import math
import numpy as np

max_value = 16

# importing training data
X_train_data = np.genfromtxt('optdigits.tra', delimiter=',') # assumes one row per image
y_train_label = np.genfromtxt('labels.csv', delimiter=',') # assumes one row per label

num_train = len(X_train_data)
X_train = X_train_data[:num_train, :] / max_value
y_train = np.zeros((num_train, 10))
y_train[np.arange(0, num_train), y_train_label[:num_train]] = 1

# importing testing data
X_test_data = np.genfromtxt('optdigits.tes', delimiter=',') # assumes one row per image
y_test_label = np.genfromtxt('labels.csv', delimiter=',') # assumes one row per label

num_test = len(X_test_data)
X_test = X_test_data[num_train:, :] / max_value
y_test = np.zeros((num_test, 10))
y_test[np.arange(0, num_test), y_test_label[y_test_label.size - num_test:]] = 1

print("Training data shape:", X_train.shape)
print("Training label shape:", y_train.shape)

batch_size = 16

weights = np.random.random(len(X_train[0])) # initialization
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
    weights = np.random.random(len(X_train[0])) # reinitialization
    weights -= .5
    print(str((N + 1) * 10) + "%" + " done training")

how_many_right = 0

for i in range(num_test):
    test = X_test[i]
    answer = np.where(y_test[i] == 1)[0]

    predicted = [[] for _ in range(10)]
    for i in range(10):
        predicted[i] = sigmoid(np.dot(test, final_weights[i]))

    guess = predicted.index(max(predicted))
    if guess == answer:
        how_many_right += 1


print(how_many_right / num_test)
np.savetxt("weights.csv", final_weights, delimiter=",")