import numpy as np
from scipy.optimize import fsolve, root
from scipy.stats import geninvgauss
from scipy.special import kn
from math import sqrt, log
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

plt.rcParams['font.family'] = 'Palatino'
plt.rcParams['font.size'] = 10

colors = ['#b7094c', '#a01a58', '#892b64', '#723c70', '#5c4d7d', '#455e89', '#2e6f95', '#1780a1', '#0091ad']
lw = 2.7
def equations(variables, n, x):
    a, b, p = variables
    if a <= 0 or b <= 0:
        return [1e6, 1e6, 1e6]
    eq1 = n*p/(2*a) + n*p*sqrt(a*b)/4 + n*b/8 * sqrt(b/a) - 1/2 * sum(x)
    eq2 = -n*p/(2*b) + n*p*sqrt(a*b)/4 + n*a/8 * sqrt(a/b) - 1/2 * sum([1/xk for xk in x])
    eq3 = n/2*log(a) - n/2 * log(b) - n * kn(p-1, sqrt(a*b))/kn(p, sqrt(a*b))*(p*(a*b-1)*(a*b)**(-3/2) + 1/2 * (p-1)*(a*b**(-1/2))) + sum([log(xk) for xk in x])
    return [eq1, eq2, eq3]


#np.random.seed(1234)
x = geninvgauss.rvs(1, 2, 1, size=5000)
noise = np.random.normal(loc=0, scale=0.3, size=2000)
# x = x + noise
x = np.sort(x)
n = len(x)

init = [2.4, 1.1, 0.7]
# a_mle, b_mle, p_mle = fsolve(equations, init, args=(n, x), xtol=1e-5, epsfcn=1e-5)
a_mle, b_mle, p_mle = root(equations, init, args=(n, x), method='lm', tol=1e-5).x
params = geninvgauss.fit(x)

print('Estimated a:', a_mle)
print('Estimated b:', b_mle)
print('Estimated p:', p_mle)

mse = np.mean(([a_mle, b_mle, p_mle] - np.array([3, 2, 1]))**2)
print('Mean Squared Error (MSE):', mse)

plt.hist(x, bins=30, density=True, alpha=0.45, label='data', color='#1e2a35')
plt.plot(x, geninvgauss.pdf(x, a_mle, b_mle, p_mle), lw=lw, label='průběh získaný metodou MLE', color=colors[0])
plt.plot(x, geninvgauss.pdf(x, 1, 2, 1), lw=lw, label='teoretický průběh', color=colors[3])
plt.plot(x, geninvgauss.pdf(x, *params), '--', lw=lw, label='automatický MLE', color=colors[5])
plt.legend(loc='best', frameon=False)
plt.show()
