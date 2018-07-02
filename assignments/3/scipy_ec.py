from scipy.optimize import minimize

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