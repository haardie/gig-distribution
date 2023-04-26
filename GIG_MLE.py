import numpy as np
from scipy.stats import geninvgauss
import matplotlib.pyplot as plt


a = 2.0
b = 1.0
p = 3.0

data = geninvgauss.rvs(a, b, p, size=1000)
noise = np.random.normal(loc=0, scale=0.1, size=1000)
data = data + noise

plt.hist(data, bins=30, density=True, alpha=0.5)

x = np.linspace(geninvgauss.ppf(0.00, a, b, p), geninvgauss.ppf(0.99, a, b, p), 1000)
plt.plot(x, geninvgauss.pdf(x, a, b, p), 'r-', lw=2, label='PDF')

plt.legend(loc='best', frameon=False)
plt.xlabel('x')
plt.ylabel('density')


params = geninvgauss.fit(data)

print('Estimated a:', params[0])
print('Estimated b:', params[1])
print('Estimated p:', params[2])

theoretical_params = [a, b, p]

mse = np.mean(([params[0], params[1], params[2]] - np.array(theoretical_params))**2)
print('Mean Squared Error (MSE):', mse)


plt.figure()
plt.hist(data, bins=30, density=True, alpha=0.5, label='data')
plt.plot(x, geninvgauss.pdf(x, a, b, p), 'r-', lw=2, label='theoretical')
plt.plot(x, geninvgauss.pdf(x, *params), 'g--', lw=2, label='estimated')
plt.legend(loc='best', frameon=False)
plt.xlabel('x')
plt.ylabel('density')
plt.show()
