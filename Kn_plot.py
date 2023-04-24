import numpy as np
from scipy.special import kn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.rcParams['text.usetex'] = True

plt.rcParams['font.family'] = 'Palatino'
plt.rcParams['font.size'] = 10

x = np.linspace(0, 6, 1000)
lw = 2.7

colors = ['#b7094c', '#a01a58', '#892b64', '#723c70', '#5c4d7d', '#455e89', '#2e6f95', '#1780a1', '#0091ad']
for N in range(6):
    plt.plot(x, kn(N, x), linewidth=lw, color=colors[N], label=r'$\mathcal{{K}}_{{{}}}(x)$'.format(N))
plt.ylim(0, 5)
plt.xlim(0, 5)
plt.xticks([1, 2, 3, 4, 5], None)
plt.xlabel(r'$x$')
plt.ylabel(r'$\mathcal{{K}}_{{{n}}}(x)$')
plt.legend()
plt.title(r'Macdonaldovy funkce $\mathcal{{K}}_{{{n}}}(x)$')
plt.savefig('Kn.png', dpi=600)
plt.show()

