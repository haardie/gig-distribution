import matplotlib.pyplot as plt
from scipy.special import kv
import numpy as np

# define the 'scaling function' and its derivative


def f(alpha, beta, lambda_):
    arg = 2 * np.sqrt(beta * lambda_)
    k = kv(alpha + 2, arg) / kv(alpha + 1, arg)
    b = np.sqrt(beta / lambda_)
    return b*k - 1  # Eq


def df(alpha, beta, lambda_):
    arg = 2 * np.sqrt(beta * lambda_)
    k = kv(alpha + 2, arg) / kv(alpha + 1, arg)
    b = np.sqrt(beta / lambda_)
    d_eq = -b * k * (alpha + 2) / lambda_ + (b * k) ** 2 - b ** 2
    return d_eq  # dEq/dlambda


# define the Newton-Raphson method
def nr_method(f, df, lambda_0, alpha, beta, max_iter, tolerance, step_size_factor):
    lambda_ = lambda_0
    lambda_vals = [lambda_]
    for i in range(max_iter):
        lambda_new = lambda_ - step_size_factor * f(alpha, beta, lambda_) / df(alpha, beta, lambda_)
        if np.abs(lambda_new - lambda_) < tolerance:
            break
        lambda_ = lambda_new
        lambda_vals.append(lambda_)

    return lambda_, lambda_vals, i + 1


# set tex interpreter and define colors
plt.rcParams['text.usetex'] = True

plt.rcParams['font.family'] = 'Palatino'
plt.rcParams['font.size'] = 10

lime = '#bee300'
colors = ['#b7094c', '#a01a58', '#892b64', '#723c70', '#5c4d7d', '#455e89', '#2e6f95', '#1780a1', '#0091ad', '#00a2b9',
          '#00b3c5']
greens = [lime, '#9fcf30', '#60cd70']

# set parameters
max_iters = 100
tol = 1e-6
step_size_factor = 0.5

alpha = -1.0
beta = 2.0
init = 1.0

domain = np.linspace(0.001, 5, 1000)

# solve the scaling equation using the Newton-Raphson method
solution, iterative_solutions, iters = nr_method(f, df, init, alpha, beta, max_iters, tol, step_size_factor)
print(f'lambda({alpha}, {beta}): {solution}, init.: {init}, iterations: {iters}')
print()

# plot the scaling function and its root

plt.plot(domain, [f(alpha, beta, lambda_) for lambda_ in domain], linewidth=2.4,
         label=r'$f(\lambda(\alpha, \beta))\equiv \sqrt{\frac{\beta}{\lambda}}\frac{\mathcal{K}_{\alpha + 1}(2\sqrt{'
               r'\beta\lambda})}{\mathcal{K}_{\alpha}(2\sqrt{\beta\lambda})} - 1 = 0$',
         color=colors[0], zorder=0)

plt.scatter(iterative_solutions, [0] * len(iterative_solutions), label='Newton-Raphsonovy iterované odhady', color=colors[4], s=40,
            zorder=1, alpha=0.4)

plt.scatter(init, 0, label=r'Počáteční volba $\lambda = \lambda_0$', color=colors[6], s=50, zorder=2,
            edgecolors=lime, linewidth=1)

plt.scatter(solution, 0, label=r'Aproximace řešení: $\lambda(\alpha, \beta) \doteq$ {:.3f}'.format(solution),
            color=lime, s=200, marker='*', zorder=3, edgecolors=colors[6], linewidth=1)

plt.axvline(solution, color='#bdbbbb', linestyle='--', linewidth=1.2, zorder=0)
plt.axhline(y=0, color='#bdbbbb', linestyle='--', linewidth=1.2, zorder=0)
plt.xlim(0, 3)
plt.ylim(-0.3, 2.0)
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$f(\lambda)$')
plt.legend()
plt.savefig('scaling_eq_sol.png', dpi=600)
plt.show()


alphas = np.linspace(-1, 5.0, 10)
betas = np.linspace(0.2, 5.0, 10)
inits = np.linspace(0.8, 6.0, 10)

lambda_nr_arr = []

for c, i in enumerate(range(len(alphas[:3]))):
    alpha = alphas[1]
    beta = betas[i]
    init = inits[i]
    solution, iterative_solutions, iters = nr_method(f, df, inits[i], alpha, beta, max_iters, tol, step_size_factor)

    print(f'lambda_{i + 1}: {solution}, init.: {inits[i]}, iterations: {iters}')

    lambda_nr_arr.append(solution)

    plt.plot(domain, [f(alpha, beta, lambda_) for lambda_ in domain], linewidth=2.4, label=r'$f(\lambda(\alpha_2, \beta_{}))$'.format(i + 1), color=colors[c],
             zorder=0)

    plt.scatter(inits[i], 0, color=greens[i], s=50, zorder=2)
    plt.scatter(lambda_nr_arr[i], 0, color=greens[i], s=200, marker='*', zorder=3, edgecolors=colors[6], linewidth=1.2)
    plt.scatter(iterative_solutions, [0] * len(iterative_solutions), color=colors[c + 4], s=30, zorder=1, alpha=0.4)
    plt.axhline(y=0, color='#bdbbbb', linestyle='--', linewidth=1.2, zorder=0)

plt.ylim(-0.2, 1.0)
plt.xlim(0.3, 2.5)
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$f(\lambda)$')
plt.legend()
plt.savefig('scaling_eq_sol_mult.png', dpi=600)
plt.show()


# compare the asymptotic behavior of the scaling function with the solution of the scaling equation
lambda_sols = []
asymptote = []

for i in range(len(betas)):
    alpha = alphas[1]
    beta = betas[i]
    init = inits[i]
    solution, iterative_solutions, iters = nr_method(f, df, inits[i], alpha, betas[i], max_iters, tol, step_size_factor)

    lambda_sols.append(solution)
    asymptote.append(beta + 3 / 2)

plt.plot(betas, asymptote, linewidth=2.4, label=r'$\lambda(\beta) = \beta + \textstyle\frac{3}{2}$', linestyle='--',
         color='#bdbbbb', zorder=0)

plt.scatter(betas, lambda_sols, s=30, label=r'$\lambda = \lambda(\alpha_2, \beta)$', color=colors[0],
            zorder=0)

plt.xlabel(r'$\beta$')
plt.ylabel(r'$\lambda(\alpha, \beta)$')
plt.ylim(0, max(lambda_sols) + 0.5)
plt.xlim(0.0, max(betas) + 0.5)
plt.legend()
plt.savefig('asympt_scaling_eq.png', dpi=600)
plt.show()

mse = np.mean((np.array(lambda_sols) - np.array(asymptote)) ** 2)
print(f'MSE: {mse}')

