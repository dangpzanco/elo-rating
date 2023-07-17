import numpy as np
import scipy.special as sp


beta_fifa = np.array([5, 60])
beta_fide = np.array([10, 40])

s_fifa = 600 / np.log(10)
s_fide = 400 / np.log(10)

# s' = s * log(10)
# beta = beta' / s'

print(f'FIFA: {beta_fifa / s_fifa}')
print(f'FIDE: {beta_fide / s_fide}')
