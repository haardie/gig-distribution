import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import geninvgauss

plt.rcParams['text.usetex'] = True

plt.rcParams['font.family'] = 'Palatino'
plt.rcParams['font.size'] = 10

colors = ['#b7094c', '#892b64', '#5c4d7d', '#455e89', '#1780a1', '#0091ad']

###
alpha = [-1, 0, 1, 3, 5, 7]
beta = 2
l = 0.5

for c, a in enumerate(alpha):
    x = np.linspace(geninvgauss.ppf(0.00, a, beta, l), geninvgauss.ppf(0.999, a, beta, l), 300)
    pdf = geninvgauss.pdf(x, a, beta, l)
    plt.xlim(0.5, 10)
    plt.ylim(0, 1.3)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')
    plt.plot(x, pdf, linewidth=2.5, label=r'$\alpha={}$'.format(a), color=colors[c])
    plt.legend(frameon=False)
plt.savefig('GIG.png', dpi=600)
plt.show()

###
alpha = 1
l = 0.5
beta = [2, 4, 6, 8]

for c, b in enumerate(beta):
    x = np.linspace(geninvgauss.ppf(0.00, alpha, b, l), geninvgauss.ppf(0.999, alpha, b, l), 300)
    pdf = geninvgauss.pdf(x, alpha, b, l)
    plt.xlim(0.5, 4)
    plt.ylim(0, 1.2)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')
    plt.xticks([1, 2, 3, 4], None)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2], None)
    plt.plot(x, pdf, linewidth=2.5, label=r'$\beta={}$'.format(b), color=colors[c])
    plt.legend(frameon=False)
# plt.title(r'Průběh hustoty GIG rozdělení pro $\alpha = 0$, $p = \textstyle\frac{1}{2}$ alpha různé hodnoty $\beta$')
plt.savefig('GIG_beta.png', dpi=600)
plt.show()
###
alpha = 1
beta = 2.0
l = [0.1, 0.5, 0.9, 1.3]

for c, l in enumerate(l):
    x = np.linspace(geninvgauss.ppf(0.00, alpha, beta, l), geninvgauss.ppf(0.999, alpha, beta, l), 300)
    pdf = geninvgauss.pdf(x, alpha, beta, l)
    plt.xlim(0, 7)
    plt.ylim(0, 0.6)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')
    plt.yticks([0, 0.2, 0.4, 0.6], None)
    plt.plot(x, pdf, linewidth=2.5, label=r'$\lambda={}$'.format(l), color=colors[c])
    plt.legend(frameon=False)
# plt.title(r'Průběh hustoty GIG rozdělení pro $\alpha = 0$, $\beta = 2$ alpha různé hodnoty $p$')
plt.savefig('GIG_p.png', dpi=600)
plt.show()

