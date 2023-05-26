import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import geninvgauss


plt.rcParams['text.usetex'] = True

plt.rcParams['font.family'] = 'Palatino'
plt.rcParams['font.size'] = 10

colors = ['#b7094c', '#892b64', '#723c70', '#455e89',  '#1780a1', '#0091ad', '#00a2b9', '#00b3c5']

p = [0, 1, 3, 5, 7]
b = 2

for i in range(len(p)):
    x = np.linspace(0.0, 15.0, 1000)
    plt.plot(x, geninvgauss.pdf(x, b, p[i]), color=colors[i], label='$p = %d$' % p[i], linewidth=2.5, zorder=10-i)
    plt.xlim(0.0, 12.0)
    plt.ylim(0.0, 0.9)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')
    plt.title(r'$b = %d$' % b)
    plt.legend(loc='best')
    plt.savefig('GIG_pdf.png', dpi=600, bbox_inches='tight')
plt.show()

