import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

ab_pairs = [(alpha, alpha) for alpha in np.arange(0,1.1,0.1)]

x = np.linspace(0, 1, 1002)[1:-1]

for a,b in ab_pairs:
    print(a,b)
    dist = beta(a, b)
    y = dist.pdf(x)
    plt.plot(x, y, label=r'$\alpha=%.1f, \beta=%.1f$'%(a,b))

plt.title(u'Bera Distribution')
plt.xlim(0, 1)
plt.ylim(0, 2.5)
plt.legend()
plt.savefig("beta.png", dpi=600)