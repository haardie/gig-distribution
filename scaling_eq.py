import matplotlib.pyplot as plt
from scipy.special import kv
import numpy as np


def f(alpha, beta, lambda_):
    arg = 2 * np.sqrt(beta * lambda_)
    return np.sqrt(beta / lambda_) * kv(alpha + 1, arg) / kv(alpha, arg) - 1


def df(alpha, beta, lambda_):
    arg = 2 * np.sqrt(beta * lambda_)
    frac_K = kv(alpha + 1, arg) / kv(alpha, arg)
    frac_B = np.sqrt(beta / lambda_)
    return frac_K * (frac_B * (1 / (2 * lambda_) + alpha / (2 * lambda_) - (alpha + 1) / lambda_) - alpha * beta / (
            lambda_ * np.sqrt(beta * lambda_))) - (frac_B * frac_K) ** 2 - frac_B ** 2


def nr_method(f, df, lambda_0, alpha, beta, max_iter, tolerance):
    lambda_ = lambda_0
    lambda_vals = [lambda_]
    for i in range(max_iter):
        lambda_new = lambda_ - f(alpha, beta, lambda_) / df(alpha, beta, lambda_)
        if np.abs(lambda_new - lambda_).any() < tolerance:
            break
        lambda_ = lambda_new
        lambda_vals.append(lambda_)

    return lambda_, lambda_vals, i + 1


alpha = -1.0
beta = 2.0
lambda0 = 1.0

lambda_nr, lambdas_nr, iterations_nr = nr_method(f, df, lambda0, alpha, beta, 500, 1e-6)
print(f'lambda({alpha}, {beta}): {lambda_nr}, init.: {lambda0}, iterations: {iterations_nr}')
print('-------------------------')

plt.rcParams['text.usetex'] = True

plt.rcParams['font.family'] = 'Palatino'
plt.rcParams['font.size'] = 10

colors = ['#b7094c', '#a01a58', '#892b64', '#723c70', '#5c4d7d', '#455e89', '#2e6f95', '#1780a1', '#0091ad', '#00a2b9',
          '#00b3c5']
lime = '#bee300'
darklime = '#6eb500'

lambdas = np.linspace(0.001, 10, 1000)
y = [f(alpha, beta, l) for l in lambdas]
plt.plot(lambdas, y, linewidth=2.4,
         label=r'$f(\lambda(\alpha, \beta))\equiv \sqrt{\frac{\beta}{\lambda}}\frac{\mathcal{K}_{\alpha + 1}(2\sqrt{'
               r'\beta\lambda})}{\mathcal{K}_{\alpha}(2\sqrt{\beta\lambda})} - 1 = 0$',
         color=colors[0], zorder=0)

plt.scatter(lambdas_nr, [0] * len(lambdas_nr), label='Newton-Raphsonovy iterované odhady', color=colors[4], s=30,
            zorder=1, alpha=0.4)

plt.scatter(lambda0, 0, label=r'Počáteční volba $\lambda = \lambda_0$', color=colors[6], s=50, zorder=2,
            edgecolors=lime, linewidth=1)

plt.scatter(lambda_nr, 0, label=r'Aproximace řešení: $\lambda(\alpha, \beta) \doteq$ {:.3f}'.format(lambda_nr),
            color=lime, s=200, marker='*', zorder=3, edgecolors=colors[6], linewidth=1)

plt.axvline(x=lambda_nr, color='#bdbbbb', linestyle='--', linewidth=1.2, zorder=0)
plt.axhline(y=0, color='#bdbbbb', linestyle='--', linewidth=1.2, zorder=0)
plt.xlim(0, 3)
plt.ylim(-0.3, 2.0)
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$f(\lambda)$')
plt.legend()
plt.savefig('scaling_eq_sol.png', dpi=600)
plt.show()

alpha_arr = np.linspace(0.5, 10.0, 10)
beta_arr = np.linspace(0.1, 10.0, 10)
lambda0_arr = np.linspace(1.0, 10.0, 10)

lambda_nr_arr = []

greens = [lime, '#9fcf30', '#60cd70']
for c, i in enumerate(range(len(alpha_arr[:3]))):
    alpha = alpha_arr[1]
    beta = beta_arr[i]
    lambda0 = lambda0_arr[i]
    lambda_nr, lambdas_nr, iterations_nr = nr_method(f, df, lambda0_arr[i], alpha, beta, 1000, 1e-6)

    print(f'lambda_{i + 1}: {lambda_nr}, init.: {lambda0_arr[1]}, iterations: {iterations_nr + 1}')

    lambda_nr_arr.append(lambda_nr)

    y_mult = [f(alpha, beta, l) for l in lambdas]

    plt.plot(lambdas, y_mult, linewidth=2.4, label=r'$f(\lambda(\alpha_1, \beta_{}))$'.format(i + 1), color=colors[c],
             zorder=0)

    plt.scatter(lambda0_arr[i], 0, color=greens[i], s=50, zorder=2)
    plt.scatter(lambda_nr_arr[i], 0, color=greens[i], s=200, marker='*', zorder=3, edgecolors=colors[6], linewidth=1.2)
    plt.scatter(lambdas_nr, [0] * len(lambdas_nr), color=colors[c + 4], s=10, zorder=1, alpha=0.4)
    # plt.axvline(x=lambda_nr_arr[i], color='#bdbbbb', linestyle='--', linewidth=1.2, zorder=0)
    plt.axhline(y=0, color='#bdbbbb', linestyle='--', linewidth=1.2, zorder=0)

plt.ylim(-0.3, 2)
plt.xlim(0, 5)
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$f(\lambda)$')
plt.legend()
plt.savefig('scaling_eq_sol_mult.png', dpi=600)
plt.show()

lambda_sols = []
asymptote = []
for i in range(len(beta_arr)):
    alpha = alpha_arr[1]
    beta = beta_arr[i]
    lambda0 = lambda0_arr[i]
    lambda_nr, lambdas_nr, iterations_nr = nr_method(f, df, lambda0_arr[i], alpha_arr[1], beta_arr[i], 1000, 1e-6)

    lambda_sols.append(lambda_nr)
    asymptote.append(beta + 3 / 2)

plt.plot(beta_arr, asymptote, linewidth=2.4, label=r'$\lambda(\beta) = \beta + \textstyle\frac{3}{2}$', linestyle='--',
         color='#bdbbbb', zorder=0)

plt.scatter(beta_arr, lambda_sols, s=50, label=r'$\lambda = \lambda(\alpha_1, \beta)$', color=colors[0],
            zorder=0)

plt.xlabel(r'$\beta$')
plt.ylabel(r'$\lambda(\alpha_1, \beta)$')
plt.ylim(0, max(lambda_sols) + 0.5)
plt.xlim(0.0, max(beta_arr) + 0.5)
plt.legend()
plt.show()

