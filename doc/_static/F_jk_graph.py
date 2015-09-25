import matplotlib.pyplot as plt
import numpy as np

def F(W, L, d):
    return d**L*(1 - d)/(1 - d**W)

W = 6
d = np.linspace(0.0, 1.0, 1000, endpoint=False)

plt.figure()
for L in xrange(0, W):
    plt.plot(d, F(W, L, d))
plt.savefig('F_jk_no_labels.svg')

