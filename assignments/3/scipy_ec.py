from scipy.optimize import minimize
import time
import numpy as np
import matplotlib.pyplot as plt
%run 'model.ipynb'

#Load MNIST training and testing images and labels
x_train = np.load("../data/mnist_tr_x.npy")
y_train = one_hot(np.load("../data/mnist_tr_y.npy"))
x_test = np.load("../data/mnist_te_x.npy")
y_test = one_hot(np.load("../data/mnist_te_y.npy"))
# visualize some of the data
print("A label looks like this: " + str(y_train[0]))
print("And an image looks like this:")
imgplot = plt.imshow(x_train[0].reshape((28,28)))

def cost(theta,X,Y):
    theta = theta.reshape((784,10))
    z = np.dot(X,theta)
    prob = softmax(z)
    cost = -1 * np.mean(np.sum(Y*np.log(prob)))
    return cost

theta = np.zeros((x_train.shape[1],y_train.shape[1]))
solution = minimize(cost,theta,args=(x_train,y_train),options={'maxiter':10})
weights = solution.x
print(weights)