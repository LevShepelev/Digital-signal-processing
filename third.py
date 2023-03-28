import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

m0, m1 = 5.0, 0.6
N = 32

def xfunc(k):
    return np.sin (2*np.pi*(m0)*k/N) + np.sin(2*np.pi*(m0+0.25)*k/N)

x1 = np.zeros((N))
for i in range (0, N):
    x1[i] = xfunc (i)

y1 = np.fft.fft (x1, N)
y2 = np.fft.fft (x1, 4*N)
visy = np.fft.fft (x1, 32*N)
print (y1)
print (y2.shape[0])

fig, axs = plt.subplots(2, 1)
fig.tight_layout (pad=2.0)

for i in [0, 1]:
    axs[i].grid(True)
    axs[i].grid(visible=True, which='major')
    axs[i].grid(visible=True, which='minor',c="#DDDDDD")
    axs[i].tick_params(axis='both', which='major', labelsize=10)
    axs[i].minorticks_on()
    axs[i].set_xlim(0, 1)
    axs[i].set_xlabel('$\\nu$', fontsize=12)
    axs[i].set_ylim(-np.max (np.absolute(visy)) - 0.5, np.max (np.absolute(visy)) + 0.5)

axs[0].set_ylabel ('N = 32', fontsize=12)
axs[1].set_ylabel ('N = 128', fontsize=12)

axs[0].plot (np.linspace (0, 1-1/(32*N-1), 32*N), np.absolute(visy[:]))
axs[0].scatter (np.linspace (0, 1-1/(N-1), N), np.absolute(y1[:]), c='tab:red',s=14)

axs[1].plot (np.linspace (0, 1-1/(32*N-1), 32*N), np.absolute(visy[:]))
axs[1].scatter (np.linspace (0, 1-1/(4*N-1), 4*N), np.absolute(y2[:]), c='tab:red',s=12)

plt.show ()