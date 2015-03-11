from numpy import *
import numpy.linalg as la
import scipy.optimize as opt
from itertools import product
import h5py
import scipy.io
import matplotlib.pyplot as plt

# Load MAT file

f = h5py.File('d2.mat','r') 
X = array(f.get('X')).T
Y = array(f.get('Y')).T
N = X.shape[0]

# Define kernel
kernel = lambda x1, x2: exp(-la.norm(x1-x2)) 	# gaussian kernel
#kernel = lambda x1, x2: dot(x1, x2) 			# linear kernel

# Precompute some matrices
YM = array(Y * Y.T)
KM = array(fromfunction(vectorize(lambda n, m: kernel(X[n,:], X[m,:])), (N, N)))

# Optimization (SLSQP)
objective = lambda a: -sum(a) + 0.5 * sum(multiply(multiply(YM, KM), matrix(a).T * matrix(a)))
constraints = [dict(type='eq', fun = lambda a: sum(matrix(a) * Y))]
result = opt.minimize(
	objective, zeros((N)), constraints=constraints,
	bounds = [(0, 1)] * N # Vary upper bound for better soft margin
)

# From here on it's only plotting ...
a = result.x
support = where(a > 1e-3)[0]

plt.figure()
plt.grid(True)

plt.plot(X[support, 0], X[support, 1], 'ok', color = [0.5, 0.5, 0.5])

indices = where(Y > 0)
plt.plot(X[indices, 0], X[indices, 1], '+r')

indices = where(Y < 0)
plt.plot(X[indices, 0], X[indices, 1], '+b')

# Plotting the decision boundary

I = 20
x1 = linspace(-2, 2, I)
x2 = linspace(-2, 2, I)
K = zeros((I,I))

b = sum(multiply(a, multiply(Y, [kernel(X[n,:], X[support[0]]) for n in range(N)])))
b = a[support[0]] * Y[support[0]] - b

for i, j in product(range(I), range(I)):
	k = [kernel(X[n,:], matrix([x1[i], x2[j]])) for n in range(N)]
	K[i,j] = sum(multiply(a, multiply(Y, k))) + b

X1, X2 = meshgrid(x1, x2)

CS = plt.contour(X1, X2, K, colors = 'k', levels=[0.0])

plt.show()
