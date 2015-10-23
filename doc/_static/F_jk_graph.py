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

plt.figure()
W = 10
L = np.arange(W)
j = L - 0.4
d = 0.8
v = np.roll(F(W, L, d), 2)
plt.bar(j, v, width=0.8, color='r', label='d=0.8')
plt.hold(True)
j = L - 0.3
d = 0.3
v = np.roll(F(W, L, d), 2)
plt.bar(j, v, width=0.6, color='b', label='d=0.3')
plt.xlim(-0.5, W - 0.5)
plt.legend()
plt.title(r'$F(l,d)$ $W=10$')
plt.xlabel('Motif state')
plt.ylabel(r'$F$')
plt.savefig('F_jk_bars.svg')
plt.savefig('F_jk_bars.pdf')

