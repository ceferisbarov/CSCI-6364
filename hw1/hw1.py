import random

import numpy as np
from matplotlib import pyplot as plt

def plot_data_and_model(X,Xtilde,Y,w,title):
	'''
	Inputs:
		X:      the original feature values  shape: [n]
		Xtilde: the design matrix            shape: [n x d+1]
		Y:      the vector of targets        shape: [n]
		w:      the parameters of the model  shape: [d+1]
	'''
	plt.figure()
	plt.scatter(X,Y)
	plt.plot(X, np.dot(Xtilde, w), color='orange')
	plt.title(title)

def polynomial_features(X,k):
	'''
	Inputs:
		X: the feature values                             shape: [n]
		k: the highest degree of the polynomial features  shape: (scalar)
	Output:
		The design matrix for degree-k polynomial features: an [n x k+1] 
      matrix whose ith row is [1, X[i], X[i]**2, X[i]**3, ..., X[i]**k]
	'''
	feature_vectors = []
	for i in range(k+1):
		temp_vector = list(map(lambda a: a**i, X))
		feature_vectors.append(temp_vector)

	design_matrix = np.array(feature_vectors).T

	return design_matrix

def solve(X, Y):
	'''
	Inputs:
		X: the feature values                       shape: [n x d+1]
		Y: labels  									shape: [n]
	Output:
		The vector W containing the solution for XT . X = XT . Y
	'''
	a = np.dot(X.T, X)
	b = np.dot(X.T, Y)
	W = np.linalg.solve(a, b)

	return W

def MSE(Y_hat, Y):
	'''
	Inputs:
		Y_hat: the predicted labels                     shape: [n x d+1]
		Y: actual labels  								shape: [n]
	Output:
		A scalar value - the squared loss.
	'''
	return np.mean((Y_hat - Y) ** 2)

def train_test_spit(X, Y, train_ratio):
	"""
	A basic implementation of train-test split without scikit-lern.
	"""
	indices = list(range(X.shape[0]))
	random.Random(3).shuffle(indices)
	
	train_count = int(train_ratio * X.shape[0])
	
	train_indices = indices[:train_count]
	test_indices = indices[train_count:]
	
	X_train, Y_train = X[train_indices], Y[train_indices]
	X_test, Y_test = X[test_indices], Y[test_indices]

	return X_train, X_test, Y_train, Y_test

def X_to_design_matrix(X):
	"""
	Adds vector of 1s to matrix X for the bias term.
	"""
	X = X.reshape(-1,1)
	ones_column = np.ones((X.shape[0], 1))
	design_matrix = np.concatenate((X, ones_column), axis=1)

	return design_matrix

X = np.load('X.npy')
Y = np.load('Y.npy')

X_train, X_test, Y_train, Y_test = train_test_spit(X, Y, train_ratio=0.8)

# ==========================================
# 					PART A
# ==========================================
X_train_tilda = X_to_design_matrix(X_train)
X_test_tilda = X_to_design_matrix(X_test)

W = solve(X_train_tilda, Y_train)
train_loss = MSE(np.dot(X_train_tilda, W), Y_train)
test_loss = MSE(np.dot(X_test_tilda, W), Y_test)

print(MSE(np.dot(X_train_tilda, W), Y_train))
print(MSE(np.dot(X_test_tilda, W), Y_test))

plot_data_and_model(X_train, X_train_tilda, Y_train, W, "Train data")
plot_data_and_model(X_test, X_test_tilda, Y_test, W, "Test data")
# ==========================================
# 					PART B
# ==========================================
# k = 20
X_train_tilda = polynomial_features(X_train, 20)
X_test_tilda = polynomial_features(X_test, 20)
W = solve(X_train_tilda, Y_train)
print(MSE(np.dot(X_train_tilda, W), Y_train))
print(MSE(np.dot(X_test_tilda, W), Y_test))

plot_data_and_model(X_train, X_train_tilda, Y_train, W, "Train data")
plot_data_and_model(X_test, X_test_tilda, Y_test, W, "Test data")
plt.figure()

# k = 1:15
train_losses = []
test_losses = []
for k in range(1,16):
	X_train_tilda = polynomial_features(X_train, k)
	X_test_tilda = polynomial_features(X_test, k)
	W = solve(X_train_tilda, Y_train)
	train_losses.append(MSE(np.dot(X_train_tilda, W), Y_train))
	test_losses.append(MSE(np.dot(X_test_tilda, W), Y_test))

plt.plot(list(range(1,16)), train_losses, label='Train Losses', color='blue')
plt.plot(list(range(1,16)), test_losses, label='Test Losses', color='orange')
plt.legend()
plt.show()
