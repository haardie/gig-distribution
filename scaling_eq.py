import matplotlib.pyplot as plt
from scipy.special import kv
import numpy as np
from scipy.optimize import curve_fit


def f(alpha, beta, lambda_):
    arg = 2 * np.sqrt(beta * lambda_)
    k = kv(alpha + 2, arg) / kv(alpha + 1, arg)
    b = np.sqrt(beta / lambda_)
    return b*k - 1


def df(alpha, beta, lambda_):
    arg = 2 * np.sqrt(beta * lambda_)
    k = kv(alpha + 2, arg) / kv(alpha + 1, arg)
    b = np.sqrt(beta / lambda_)
    d_eq = -b * k * (alpha + 2) / lambda_ + (b * k) ** 2 - b ** 2
    return d_eq


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


def plot_nr_solution(dom, f, init_guess, iter_sols, sol, f_color, iter_color, init_color, sol_color, f_label=None, init_label=None, iter_label=None, sol_label=None):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'Palatino'
    plt.rcParams['font.size'] = 10
    plt.plot(dom, f, linewidth=2.4, label=f_label, color=f_color, zorder=0)
    plt.scatter(iter_sols, [0] * len(iter_sols), label=iter_label, color=iter_color, s=40, zorder=1, alpha=0.4)
    plt.scatter(init_guess, 0, label=init_label, color=init_color, s=50, zorder=2)
    plt.scatter(sol, 0, label=sol_label, color=sol_color, s=200, marker='*', zorder=3, edgecolors=colors_cold[6], linewidth=1)
    plt.axvline(sol, color='#bdbbbb', linestyle='--', linewidth=1.2, zorder=0)
    plt.axhline(y=0, color='#bdbbbb', linestyle='--', linewidth=1.2, zorder=0)
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$f(\lambda)$')
    plt.legend()
    pass


def plot_scattered_data(dom, points, theor_f, points_label, theor_f_label, xlabel, ylabel, xmin, xmax, ymin, ymax):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'Palatino'
    plt.rcParams['font.size'] = 10
    plt.scatter(dom, points, label=points_label, color=colors_cold[0], s=40, zorder=1)
    plt.plot(dom, theor_f, label=theor_f_label, color='#bdbbbb', linewidth=2.4, linestyle='--', zorder=0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.legend()
    pass

def exp_function(x, a, b, c):
    return a * np.exp(-b * x) + c


colors_cold = ['#b7094c', '#a01a58', '#892b64', '#723c70', '#5c4d7d', '#455e89', '#2e6f95', '#1780a1', '#0091ad', '#00a2b9', '#00b3c5']
colors_green = ['#bee300', '#9fcf30', '#60cd70']

# set parameters
max_iters = 100
tol = 1e-6
step_size_factor = 0.5

alpha = float(input(f'Enter the value of alpha: '))
beta = float(input(f'Enter the value of beta: '))
init = float(input(f'Enter the initial guess for lambda: '))
print()

domain = np.linspace(0.001, 10, 1000)

# solve the scaling equation using the Newton-Raphson method
solution, iterative_solutions, iters = nr_method(f, df, init, alpha, beta, max_iters, tol, step_size_factor)
print(f'lambda({alpha}, {beta}): {solution}, init.: {init}, iterations: {iters}')
print()

plot_nr_solution(dom=domain,
                 f=[f(alpha, beta, lambda_) for lambda_ in domain], init_guess=init,
                 iter_sols=iterative_solutions,
                 sol=solution,
                 f_color=colors_cold[0],
                 iter_color=colors_cold[4],
                 init_color=colors_cold[6],
                 sol_color=colors_green[0],
                 f_label=r'$f(\lambda(\alpha, \beta)) = \sqrt{\frac{\beta}{\lambda}}\frac{\mathcal{K}_{\alpha + 2}(2\sqrt{\beta\lambda})}{\mathcal{K}_{\alpha + 1}(2\sqrt{\beta\lambda})} - 1$',
                 init_label=r'Počáteční volba $\lambda = \lambda_0$', iter_label='Newton-Raphsonovy iterované odhady',
                 sol_label=r'Numerické řešení $\lambda(\alpha, \beta)$')
plt.xlim(min(init, solution) - 1, max(init, solution) + 1)
plt.ylim(-0.2, f(alpha, beta, min(init, solution) - 1) + 1)
plt.savefig('scaling_eq_sol.png', dpi=600)
plt.show()

betas = np.linspace(0.2, 5.0, 10)
inits = np.linspace(1.0, 6.0, 10)

solutions = []
for c, i in enumerate(range(3)):
    beta = betas[i]
    init = inits[i]
    solution, iterative_solutions, iters = nr_method(f, df, inits[i], alpha, beta, max_iters, tol, step_size_factor)

    print(f'lambda_{i + 1}: {solution:.3f}, init.: {inits[i]:.3f}, iterations: {iters}')

    solutions.append(solution)

    plot_nr_solution(dom=domain,
                     f=[f(alpha, beta, lambda_) for lambda_ in domain],
                     init_guess=inits[i],
                     iter_sols=iterative_solutions,
                     sol=solutions[i],
                     f_color=colors_cold[c],
                     iter_color=colors_cold[c + 4],
                     init_color=colors_green[i],
                     sol_color=colors_green[i],
                     f_label=r'$f(\lambda(\alpha, \beta_{}))$'.format(i + 1))

plt.xlim(min(inits[0], solutions[0]) - 0.5, max(solutions[2], inits[2]) + 0.5)
plt.ylim(-0.2, f(alpha, betas[2], min(inits[2], solutions[2]) - 0.5) + 0.5)
plt.savefig('scaling_eq_sol_mult.png', dpi=600)
plt.show()


# compare the asymptotic behavior of the scaling function with the solution of the scaling equation
solutions2 = []
asymptote = []

for i in range(len(betas)):
    beta = betas[i]
    init = inits[i]
    solution, iterative_solutions, iters = nr_method(f, df, inits[i], alpha, betas[i], max_iters, tol, step_size_factor)

    solutions2.append(solution)
    asymptote.append(alpha + beta + 1.5)


solution_fit = np.polyfit(betas, solutions2, 1)
print('\n Solution fit:')
print(f'lambda(beta) = {solution_fit[0]:.3f} * beta + {solution_fit[1]:.3f}')

plot_scattered_data(dom=betas,
                    points=solutions2,
                    theor_f=asymptote,
                    points_label=r'$\lambda = \lambda(\alpha, \beta)$',
                    theor_f_label=r'$\lambda_a(\alpha, \beta) = \alpha + \beta + \textstyle\frac{3}{2}$',
                    xlabel=r'$\beta$', ylabel=r'$\lambda(\alpha, \beta)$',
                    xmin=0.0, xmax=max(betas) + 0.5,
                    ymin=min(solutions2) - 1, ymax=max(solutions2) + 0.5)
plt.savefig('asympt_scaling_eq.png', dpi=600)
plt.show()


mse = np.mean((np.array(solutions2) - np.array(asymptote)) ** 2)
print('\n Asymptotic behaviour vs solution:')
print(f'MSE: {mse:.3f}')

residuals = np.abs(np.array(solutions2) - np.array(asymptote))
params, params_cov = curve_fit(exp_function, betas, residuals)
intercept, scale, decay = params
residuals_fit = exp_function(betas, intercept, scale, decay)
print('\n Residuals fit:')
print(f'r(beta) = {intercept:.3f} + {scale:.3f} * exp(-{decay:.3f} * beta)')
plot_scattered_data(dom=betas,
                    points=residuals,
                    theor_f=residuals_fit,
                    points_label=r'$r = |\lambda(\alpha, \beta) - \lambda_a(\alpha, \beta)|$',
                    theor_f_label=r'$f(\beta) = {} + {} e^{{-{} \cdot \beta}}$'.format(round(intercept, 2), round(scale, 2), round(decay, 2)),
                    xlabel=r'$\beta$',
                    ylabel=r'$r$',
                    xmin=0.0, xmax=max(betas) + 0.5,
                    ymin=min(residuals) - 0.1, ymax=max(residuals) + 0.1)
plt.savefig('residuals_scaling_eq.png', dpi=600)
plt.show()
