import matplotlib.pyplot as plt
import scipy
import numpy as np
from scipy.optimize import curve_fit


# defines the scaling equation
def f(alpha, beta, lambda_):
    arg = 2 * np.sqrt(beta * lambda_)
    k = scipy.special.kv(alpha + 2, arg) / scipy.special.kv(alpha + 1, arg)
    b = np.sqrt(beta / lambda_)
    return b*k - 1

# computes the derivative of the scaling equation
def df(alpha, beta, lambda_):
    arg = 2 * np.sqrt(beta * lambda_)
    k = scipy.special.kv(alpha + 2, arg) / scipy.special.kv(alpha + 1, arg)
    b = np.sqrt(beta / lambda_)
    d_eq = -b * k * ((alpha + 2) / lambda_) + (b * k) ** 2 - b ** 2
    return d_eq


# performs Newton-Raphson method
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


# finds the constraint on alpha and beta for the scaling equation to be solvable
def get_solvability_constraint(alpha, max_iter, tolerance, step_size_factor):
    const = 2.0
    step = 0.1
    if alpha < -2:
        while True:
            beta = -alpha - const
            lambda_0 = np.abs(1.5 - const)
            if beta <= 0:
                print(f'beta = {beta}. The scaling equation is not solvable for nonpositive beta.')
                return None
            else:
                # when lambda_ is either nan or near zero, the Newton-Raphson method fails to converge
                # silence the warning
                np.seterr(divide='ignore', invalid='ignore')
                lambda_, _, _ = nr_method(f, df, lambda_0, alpha, beta, max_iter, tolerance, step_size_factor)
                if isinstance(lambda_, float) and lambda_ > 0:  # if the solution is a number and is positive
                    const += step
                    step *= 0.1     # decrease the step size (increase the precision of the condition)

                    if step < 1e-5:  # when reaching the 'edge' of float precision, stop
                        print(f'The solvability constraint is: alpha + beta + {const:.5f} > 0')
                        print(f'Choose beta > {-alpha - const:.5f}')
                        break
                else:
                    const -= step

    else:
        print(f'alpha = {alpha:.3f} >= -2. The scaling equation is solvable for any beta > 0.')
    return const


# finds the initial guess for the Newton-Raphson method
def get_initial_guess(alpha, beta):
    initial_guess = alpha + beta + 1.0
    return initial_guess


# plots the solution of the scaling equation
def plot_nr_solution(dom, f, init_guess, iter_sols, sol, f_color, iter_color, init_color, sol_color, f_label=None, 
                     init_label=None, iter_label=None, sol_label=None):
    
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'Palatino'
    plt.rcParams['font.size'] = 10
    plt.plot(dom, f, linewidth=2.4, label=f_label, color=f_color, zorder=0)
    plt.scatter(iter_sols, [0] * len(iter_sols), label=iter_label, color=iter_color, s=40, zorder=1, alpha=0.4)
    plt.scatter(init_guess, 0, label=init_label, color=init_color, s=50, zorder=2)
    plt.scatter(sol, 0, label=sol_label, color=sol_color, s=200, marker='*', zorder=3, 
                edgecolors=colors_cold[6], linewidth=1)
    plt.axvline(sol, color='#bdbbbb', linestyle='--', linewidth=1.2, zorder=0)
    plt.axhline(y=0, color='#bdbbbb', linestyle='--', linewidth=1.2, zorder=0)
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$F(\lambda)$')
    plt.legend(loc='upper right', frameon=False)
    pass


# plots scattered data (residuals, etc.)
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
    plt.legend(loc='upper right', frameon=False)
    pass


# is used to fit the residuals
def exp_function(x, a, b, c):
    x = np.array(x)
    return a * np.exp(-b * x) + c


# defines GIG pdf
def gig_pdf(x, alpha, beta, lambda_):
    x = np.array(x)
    b = beta/lambda_
    arg = 2 * np.sqrt(beta * lambda_)
    k = scipy.special.kv(alpha + 1, arg)
    pdf = 0.5*(b**(-(alpha+1)/2))*(1/k)*(x**(alpha+1))*np.exp(-(beta/x + lambda_*x)/2)
    return pdf


colors_cold = ['#b7094c', '#a01a58', '#892b64', '#723c70', '#5c4d7d', '#455e89', 
               '#2e6f95', '#1780a1', '#0091ad', '#00a2b9', '#00b3c5']
colors_green = ['#bee300', '#9fcf30', '#60cd70', '#88b400']


max_iters = 100
tol = 1e-6
step_size_factor = 0.5
domain = np.linspace(0.001, 10, 1000)
inits = np.linspace(0.1, 5.0, 10)

print('Solve for a sigle pair (alpha, beta):')
print()

alpha = float(input(f'Enter the value of alpha: '))
constraint = get_solvability_constraint(alpha, max_iters, tol, step_size_factor)
beta = float(input(f'Enter the value of beta: \n'))
init_ = get_initial_guess(alpha, beta)

solution, iterative_solutions, iters = nr_method(f, df, init_, alpha, beta, max_iters, tol, step_size_factor)
print(f'lambda({alpha}, {beta}): {solution:.3f}, init.: {init_:.3f}, iterations: {iters}')
print()

plot_nr_solution(dom=domain,
                 f=[f(alpha, beta, lambda_) for lambda_ in domain], init_guess=init_,
                 iter_sols=iterative_solutions,
                 sol=solution,
                 f_color=colors_cold[0],
                 iter_color=colors_cold[4],
                 init_color=colors_cold[6],
                 sol_color=colors_green[0],
                 f_label=r'$F(\lambda(\alpha, \beta)) = \sqrt{\frac{\beta}{\lambda}}\frac{\mathcal{K}_{\alpha + 2}(2\sqrt{\beta\lambda})}{\mathcal{K}_{\alpha + 1}(2\sqrt{\beta\lambda})} - 1$',
                 init_label=r'Počáteční volba $\lambda = \lambda_0$', iter_label='Newtonovy-Raphsonovy iterované odhady',
                 sol_label=r'Numerické řešení $\lambda(\alpha, \beta)$')
if alpha < -2:
    plt.xlim(0.0, max(init_, solution) + 0.2)
    plt.ylim(-0.2, solution + 0.5)
else:
    plt.xlim(min(init_, solution) - 1, max(init_, solution) + 1)
    plt.ylim(-0.2, f(alpha, beta, min(init_, solution)) + 0.2)

plt.title(r'$\alpha = {:.0f}, \beta = {:.0f}$'.format(alpha, beta))
plt.savefig('scaling_eq_sol.png', dpi=600, bbox_inches='tight')
plt.show()

print('Solve for a range of (alpha, beta):')
solutions = []
inits_ = []

# alpha = np.random.uniform(-3.0, 3.0)
alpha = -1.412
for c, i in enumerate(range(3)):
    const = get_solvability_constraint(alpha, max_iters, tol, step_size_factor)

    if alpha < -2:
        beta = -alpha - const + (i+1) * 0.5

    else:
        beta = i+1

    print(f'beta_{i + 1}: {beta:.3f}')
    init_ = get_initial_guess(alpha, beta)
    inits_.append(init_)

    solution, iterative_solutions, iters = nr_method(f, df, init_, alpha, beta, max_iters, tol, step_size_factor)

    print(f'lambda_{i + 1}: {solution:.3f}, init.: {init_:.3f}, iterations: {iters}')
    print()

    solutions.append(solution)

    plot_nr_solution(dom=domain,
                     f=[f(alpha, beta, lambda_) for lambda_ in domain],
                     init_guess=init_,
                     iter_sols=iterative_solutions,
                     sol=solutions[i],
                     f_color=colors_cold[c],
                     iter_color=colors_cold[c + 4],
                     init_color=colors_green[i],
                     sol_color=colors_green[i],
                     f_label=r'$F(\lambda(\alpha, \beta_{}))$'.format(i + 1))

plt.xlim(min(min(inits_), min(solutions)) - 0.5, max(max(solutions), max(inits_)) + 0.5)
plt.ylim(-0.2, f(alpha, beta, min(min(inits_), min(solutions))) + 1)
plt.title(r'$\alpha \doteq -1.4,\hspace{0.8mm} \beta_1 = 1,\hspace{0.8mm} \beta_2 = 2, \hspace{0.8mm} \beta_3 = 3$')
plt.savefig('scaling_eq_sol_mult.png', dpi=600, bbox_inches='tight')
plt.show()

# compare the asymptotic behavior of the scaling function with the solution of the scaling equation
print('Asymptotic behavior vs the solution')
print()

if alpha < -2:
    c = get_solvability_constraint(alpha, max_iters, tol, step_size_factor)
    betas = [-alpha - c + (i+1)*0.2 for i in range(10)]
else:
    betas = np.linspace(1, 10, 10)

solutions2 = []
asymptote = []

for i in range(len(betas)):
    beta = betas[i]
    init_ = get_initial_guess(alpha, beta)
    solution, iterative_solutions, iters = nr_method(f, df, init_, alpha, betas[i], max_iters, tol, step_size_factor)

    solutions2.append(solution)
    asymptote.append(alpha + beta + 1.5)

solution_fit = np.polyfit(betas, solutions2, 1)
print('\n Solution fit:')
print(f'lambda(alpha, beta) = {solution_fit[0]:.3f} * beta + alpha + {(solution_fit[1] - alpha):.3f}')

plot_scattered_data(dom=betas,
                    points=solutions2,
                    theor_f=asymptote,
                    points_label=r'$\lambda = \lambda(\alpha, \beta)$',
                    theor_f_label=r'$\lambda_a(\alpha, \beta) = \alpha + \beta + \textstyle\frac{3}{2}$',
                    xlabel=r'$\beta$', ylabel=r'$\lambda(\alpha, \beta)$',
                    xmin=min(betas) - 0.1, xmax=max(betas) + 0.1,
                    ymin=min(solutions2) - 0.2, ymax=max(betas) + alpha + 1.5 + 0.5)
plt.savefig('asympt_scaling_eq.png', dpi=600, bbox_inches='tight')
plt.show()

mse = np.mean((np.array(solutions2) - np.array(asymptote)) ** 2)
print(f'MSE: {mse:.3f}')

# compute and plot the residuals
residuals = np.abs(np.array(solutions2) - np.array(asymptote))/np.array(asymptote)
params, _ = curve_fit(exp_function, betas, residuals)
intercept, scale, decay = params
residuals_fit = exp_function(betas, intercept, scale, decay)
print('\n Residuals fit:')
print(f'r(beta) = {intercept:.3f} + {scale:.3f} * exp(-{decay:.3f} * beta)')
plot_scattered_data(dom=betas,
                    points=residuals,
                    theor_f=residuals_fit,
                    points_label=r'$r = |(\lambda(\alpha, \beta) - \lambda_a(\alpha, \beta))\cdot\lambda_a^{-1}|$',
                    theor_f_label=r'$R(\beta) = {} + {} e^{{-{} \cdot \beta}}$'.format(round(intercept, 2), round(scale, 2), round(decay, 2)),
                    xlabel=r'$\beta$',
                    ylabel=r'$r$',
                    xmin=min(betas) - 0.1, xmax=max(betas) + 0.1,
                    ymin=0.0, ymax=max(residuals) + 0.02)
plt.savefig('residuals_scaling_eq.png', dpi=600, bbox_inches='tight')
plt.show()

# compare the GIG pdf for asymptotic lambda with the NR solution
x = np.linspace(0.001, 5.0, int(1e2))
alpha = -3.0
beta = 3.0
init_ = get_initial_guess(alpha, beta)
lambda_, _, _ = nr_method(f, df, init_, alpha, beta, max_iters, tol, step_size_factor)
pdf = gig_pdf(x, alpha, beta, lambda_)
pdf_asympt = gig_pdf(x, alpha, beta, alpha + beta + 1.5)

auc = np.trapz(pdf, x=x)    # area under the curve
auc_asympt = np.trapz(pdf_asympt, x=x)

abc = np.abs(auc - auc_asympt)  # area between the curves

print(f'Asymptotic AUC: {auc_asympt:.3f}')
print(f'NR solution AUC: {auc:.3f}')
print(f'ABC: {abc:.3f}')

plt.plot(x, pdf, label=r'$f(x, \alpha, \beta, \lambda)$', linewidth=2.4, color=colors_green[0])
plt.plot(x, pdf_asympt, label=r'$f(x, \alpha, \beta, \lambda_a)$', linewidth=2.4, color=colors_cold[0])
plt.fill_between(x, pdf, pdf_asympt, color=colors_cold[7], alpha=0.6, hatch='//', 
                 edgecolor=colors_cold[6], linewidth=0.7)
plt.text(1.2, 11.0, r'$ABC \doteq {}$'.format(round(abc, 2)), fontsize=12, color=colors_cold[6])
plt.xlabel(r'$x$')
plt.ylabel(r'$f(x, \alpha, \beta, \lambda)$')
plt.legend(frameon=False, fontsize=10)
plt.xlim(0.0, 5.0)
plt.ylim(0.0, max(pdf) + 0.5)
plt.savefig('gig_pdf_scaling_eq.png', dpi=600, bbox_inches='tight')
plt.show()


if auc > 1:
    print('Your are not dealing with a pdf. Normalization triggered.')

    pdf = pdf / auc
    pdf_asympt = pdf_asympt / auc_asympt
    abc = np.abs(np.trapz(pdf - pdf_asympt, x=x))

plt.plot(x, pdf, label=r'$f(x, \alpha, \beta, \lambda)$', linewidth=2.4, color=colors_green[0])
plt.plot(x, pdf_asympt, label=r'$f(x, \alpha, \beta, \lambda_a)$', linewidth=2.4, color=colors_cold[0])
plt.fill_between(x, pdf, pdf_asympt, color=colors_cold[7], alpha=0.6, hatch='//', 
                 edgecolor=colors_cold[6], linewidth=0.7)
plt.text(1.2, 0.8, r'$ABC \doteq {:.2e}$'.format(abc), fontsize=12, color=colors_cold[6])
plt.xlabel(r'$x$')
plt.ylabel(r'$f(x, \alpha, \beta, \lambda)$')
plt.legend(frameon=False, fontsize=10)
plt.xlim(0.0, 5.0)
plt.ylim(0.0, max(pdf) + 0.05)
plt.savefig('gig_pdf_norm_scaling_eq.png', dpi=600, bbox_inches='tight')
plt.show()
