from numpy import *
import scipy.optimize as opt
from itertools import product
import numpy.linalg as la
import scipy.linalg as sla

x0 = array([10, 10, 10])

objective = lambda x: 0.5 * x[0]**2 + 0.5 * x[1] ** 2

constraints = [
	dict(type = 'ineq', fun = lambda x: 2*x[0] + 2*x[1] + x[2] - 1),
	dict(type = 'ineq', fun = lambda x: 4*x[0] + 4*x[1] + x[2] - 1),
	dict(type = 'ineq', fun = lambda x: 4*x[0] + x[2] - 1),
	dict(type = 'ineq', fun = lambda x: -x[2] - 1),
	dict(type = 'ineq', fun = lambda x: -2*x[0] - x[2] - 1),
	dict(type = 'ineq', fun = lambda x: -2*x[1] - x[2] - 1)
]

result = opt.minimize(objective, x0, constraints=constraints)
print('Primal Result', result.x)
print()

import sympy as sp
w1, w2, b = sp.symbols('w1, w2, b')
a = sp.symbols('a0:6')

objective = 0.5 * w1**2 + 0.5 * w2**2
constraints = [
	2*w1 + 2*w2 + b - 1,
	2*w1 + 4*w2 + b - 1,
	4*w1 + b - 1,
	-b - 1,
	-2*w1 - b - 1,
	-2*w2 - b - 1
]

lagrange = objective + sum([-a[i] * constraints[i] for i in range(6)])
print('L = ', lagrange)
print()

dw1 = lagrange.diff(w1)
dw2 = lagrange.diff(w2)
db = lagrange.diff(b)

print('dL/dw1 = ', dw1)
print('dL/dw2 = ', dw2)
print('dL/db  = ', db)
print()

w1_ = sp.solve(dw1, w1)[0]
w2_ = sp.solve(dw2, w2)[0]

print('w1 = ', w1_)
print('w2 = ', w2_)
print()

dual = lagrange.subs({w1 : w1_, w2 : w2_}).simplify()
dual = dual.subs({b:0}).simplify()

print('d = ', dual)
print()

objective_ = sp.lambdify(a, dual)
objective = lambda a: -objective_(*a)

a0 = [0] * 6
bounds = [(0, None) for i in range(6)]
constraints = [dict(type='eq', fun=lambda a: sum(-a[0:3] + a[3:6]))]

result = opt.minimize(objective, a0, bounds=bounds, constraints=constraints)
print('Dual Result', result.x)
print()

print('w1 = ', w1_.subs({a[i]: result.x[i] for i in range(6)}))
print('w2 = ', w2_.subs({a[i]: result.x[i] for i in range(6)}))
print()

a_ = sp.solve(db, a[5])[0]
dual = dual.subs({a[5]: a_})
dual = dual.simplify()

da = [dual.diff(a[i]) for i in range(6)]
[print('dd/da%d = ' % i, da[i]) for i in range(6)]
print()
