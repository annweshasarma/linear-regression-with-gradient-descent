import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



data = pd.read_csv('train.csv')
x = data['GrLivArea']
y = data['SalePrice']

x = (x - x.mean()) / x.std()
x = np.c_[np.ones(x.shape[0]), x] 

alpha = 0.01
iterations = 2000
m=y.size 
np.random.seed(123)
theta = np.random.rand(2) 

def gradient_descent(x, y, theta, iterations, alpha):
    past_costs = []
    past_thetas = [theta]
    for i in range(iterations):
        prediction = np.dot(x, theta)
        error = prediction - y
        cost = 1/(2*m) * np.dot(error.T, error)
        past_costs.append(cost)
        theta = theta - (alpha * (1/m) * np.dot(x.T, error))
        past_thetas.append(theta)
       
    return past_thetas, past_costs
    

past_thetas, past_costs = gradient_descent(x, y, theta, iterations, alpha)
theta = past_thetas[-1]


print("Gradient Descent: {:.2f}, {:.2f}".format(theta[0], theta[1]))

plt.title('Cost Function J')
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
plt.plot(past_costs)
plt.show()


