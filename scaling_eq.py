import matplotlib.pyplot as plt
from scipy.special import kv
from scipy.optimize import newton

import numpy as np
from scipy.misc import derivative


def f(alpha, beta, lambda_):
    arg = 2 * np.sqrt(beta * lambda_)
    return np.sqrt(beta/lambda_)*kv(alpha+1, arg)/kv(alpha, arg) - 1


def df(alpha, beta, lambda_):
    arg = 2*np.sqrt(beta*lambda_)
    frac_K = kv(alpha+1, arg)/kv(alpha, arg)
    frac_B = np.sqrt(beta/lambda_)
    return frac_K*(frac_B*(1/(2*lambda_) + alpha/(2*lambda_) - (alpha + 1)/lambda_) - alpha*beta/(lambda_*np.sqrt(beta*lambda_))) - (frac_B*frac_K)**2 - frac_B**2


def nr_method(f, df, lambda_0, alpha, beta, max_iter, tolerance):
    lambda_ = lambda_0
    lambda_vals = [lambda_]
    for i in range(max_iter):
        lambda_new = lambda_ - f(alpha, beta, lambda_)/df(alpha, beta, lambda_)
        if np.abs(lambda_new - lambda_).any() < tolerance:
            break
        lambda_ = lambda_new
        lambda_vals.append(lambda_)

    return lambda_, lambda_vals, i+1

alpha = -1.0
beta = 2.0
lambda0 = 1.0

lambda_nr, lambdas_nr, iterations_nr = nr_method(f, df, lambda0, alpha, beta, 1000, 1e-8 )
print(f'Newton-Raphson lambda: {lambda_nr}, iterations: {iterations_nr}')


plt.rcParams['text.usetex'] = True

plt.rcParams['font.family'] = 'Palatino'
plt.rcParams['font.size'] = 10


colors = ['#b7094c', '#a01a58', '#892b64', '#723c70', '#5c4d7d', '#455e89', '#2e6f95', '#1780a1', '#0091ad', '#00a2b9', '#00b3c5']
lime = '#BEE300'
darklime = '#6EB500'



lambdas = np.linspace(0.01, 3.0, 1000)
y = [f(alpha, beta, l) for l in lambdas]
plt.plot(lambdas, y, linewidth=2.4, label=r'$f(\lambda(\alpha, \beta))\equiv \sqrt{\frac{\beta}{\lambda}}\frac{\mathcal{K}_{\alpha + 1}(2\sqrt{\beta\lambda})}{\mathcal{K}_{\alpha}(2\sqrt{\beta\lambda})} - 1 = 0$', color=colors[0], zorder=0)
plt.scatter(lambdas_nr, [0]*len(lambdas_nr), label='Newton-Raphsonovy iterované odhady', color=colors[4], s=30, zorder=1, alpha=0.8)
plt.scatter(lambda0, 0, label=r'Počáteční volba $\lambda = \lambda_0$', color=colors[6], s=50, zorder=2, edgecolors=lime, linewidth=1)
plt.scatter(lambda_nr, 0, label=r'Aproximace řešení: $\lambda(\alpha, \beta) \doteq$ {:.3f}'.format(lambda_nr), color=lime, s=200, marker='*', zorder=3, edgecolors=colors[6], linewidth=1)
plt.axvline(x=lambda_nr, color='#BDBBBB', linestyle='--', linewidth=1.2, zorder=0)
plt.axhline(y=0, color='#BDBBBB', linestyle='--', linewidth=1.2, zorder=0)
plt.xlim(0, 2.5)
plt.ylim(-0.3, 2)
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$f(\lambda)$')
plt.legend()
plt.savefig('scaling_eq_sol.png', dpi=600)
plt.show()
