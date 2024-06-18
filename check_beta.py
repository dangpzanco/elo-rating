import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import model_utils as mutils
import plot_utils as lutils


M = np.array([5, 15, 20, 30, 200])
v = np.logspace(-2, 2, 1000)

fig, ax = plt.subplots()
for m in M:
    K = m * (m - 1)
    # K = 10 * (m - 1)
    beta_opt = mutils.optimal_beta_k(v, m, K, hfa=0)
    ax.plot(v, beta_opt, label=f'M = {m}')
ax.plot(v, v, color='k', linestyle='--', label='v')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Variance $v$')
ax.set_ylabel('Optimum step size')
ax.legend()
plt.show()

print()


