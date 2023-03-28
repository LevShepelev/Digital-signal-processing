import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

m0, m1 = 5.0, 0.25
N = 32

def dtft(posx, negx, nu):
   sum = 0
   if posx is not None:
       for i in range (posx.shape[0]):
           sum += posx[i] * np.exp (-1j * 2 * np.pi * nu * i)

   if negx is not None:
       for i in range (negx.shape[0]):
           sum += negx[i] * np.exp (1j * 2 * np.pi * nu * (i+1))

   return sum


def xfunc(k, m):
    return np.sin (2*np.pi*(m)*(k%N)/N) + np.cos(2*np.pi*m*(k%N)/N)

x1 = np.zeros((N))
for i in range (0, N):
    x1[i] = xfunc (i, m0)

x2 = np.zeros((N))
for i in range (0, N):
    x2[i] = xfunc (i, m0+m1)

y1 = np.fft.fft (xfunc (np.arange (0, N), m0), N) / (N)
y2 = np.fft.fft (xfunc (np.arange (0, N), m0+m1), N) / (N)

vy1 = np.fft.fft (xfunc (np.arange (0, 32*N), m0), 32*N) / (32*N)
vy2 = np.fft.fft (xfunc (np.arange (0, 32*N), m0+m1), 32*N) / (32*N)
fig, axs = plt.subplots(2, 1)
fig.tight_layout (pad=2.0)

Nd = 1024
N = 32
x = np.zeros((Nd), dtype=complex)
for k in range(0, N):
    x[k] = np.cos(2 * np.pi * (m0) * k / N) + np.sin(2 * np.pi * (m0) * k / N)
y = np.fft.fft (x, Nd) / N

x3 = np.zeros((Nd), dtype=complex)
for p in range(0, N):
    x3[p] = np.cos(2 * np.pi * (m0 + m1) * (p % N) / N) + np.sin(2 * np.pi * (m0 + m1) * (p % N) / N)
y3 = np.fft.fft (x3, Nd) / N


for i in [0, 1]:
    axs[i].grid(True)
    axs[i].grid(visible=True, which='major')
    axs[i].grid(visible=True, which='minor',c="#DDDDDD")
    axs[i].tick_params(axis='both', which='major', labelsize=10)
    axs[i].minorticks_on()
    axs[i].set_xlim(0, 1)
    axs[i].set_xlabel('$\\nu$', fontsize=12)

axs[0].set_ylim(-0.2, np.max (np.absolute(vy1)) + 0.5)
axs[1].set_ylim(-0.2, np.max (np.absolute(vy2)) + 0.5)

axs[0].set_ylabel ('m = 5', fontsize=12)
axs[1].set_ylabel ('m = 5.6', fontsize=12)

axs[0].plot (np.linspace (0, 1-1/(32*N-1), 32*N), np.absolute(vy1[:]))
axs[0].plot(np.linspace (0, 1-1/(32*N-1), 32*N), np.absolute(y))
axs[0].scatter (np.linspace (0, 1-1/(N-1), N), np.absolute(y1[:]), c='tab:red',s=14)

axs[1].plot (np.linspace (0, 1-1/(32*N-1), 32*N), np.absolute(vy2[:]))
axs[1].plot(np.linspace (0, 1-1/(32*N-1), 32*N), np.absolute(y3))
axs[1].scatter (np.linspace (0, 1-1/(N-1), N), np.absolute(y2[:]), c='tab:red',s=14)

plt.show ()