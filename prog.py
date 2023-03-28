import matplotlib.pyplot as plt
import numpy as np

task = 1


def dtft(posx, negx, nu):
   sum = 0
   if posx is not None:
       for i in range (posx.shape[0]):
           sum += posx[i] * np.exp (-1j * 2 * np.pi * nu * i)

   if negx is not None:
       for i in range (negx.shape[0]):
           sum += negx[i] * np.exp (1j * 2 * np.pi * nu * (i+1))

   return sum

def convol(x, y):
   sum = np.zeros(x.shape[0], dtype=complex)
   for i in range(x.shape[0]):
       for k in range(i, sum.shape[0]):
           sum[k] += x[i] * y[k - i]
   return sum

Nd = 1500
N = 8
L = 4 #1 variant
nu0 = 0.1

x = np.zeros ((Nd), dtype=complex)

if task == 1:
    x[0:N] = 1
if task == 2:
    for i in range(0, N*L, L):
        x[i] = 1
if task == 3:
    x[0:N]=1
    for i in range (N):
        x[i] *= i
if task == 4:
    for i in range (N):
        x[i] = np.exp (2 * np.pi * nu0 * i * 1j)
if task == 5:
    x[0:N] = 1
    #x = convol (x, x)
    x = np.convolve (x, x)
    x = x[0:Nd]

y = np.fft.fftshift(np.fft.fft (x, Nd))

#y = dtft(x, None ,np.linspace (-0.5, 0.5, Nd))

yph = np.angle (y)
yabs = np.absolute (y)
yre = np.real(y)
yim = np.imag(y)

if (task == 1 or task == 3):
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
else:
    fig, (ax0, ax1) = plt.subplots(2, 1)

fig.tight_layout (pad=2.0)

ax0.stem(range (0, np.nonzero(x)[0][-1]+4), np.real(x[0:np.nonzero(x)[0][-1]+4]))
markerline, stemline = ax0.stem(range (0, np.nonzero(x)[0][-1]+4), np.imag(x[0:np.nonzero(x)[0][-1]+4]))[0:2]
markerline.set_color('purple')
stemline.set_color('purple')
ax0.set_xlim(-0.1, np.nonzero(x)[0][-1]+3)
ax0.set_ylim(-np.max (np.absolute(x)) - 0.5, np.max (np.absolute(x)) + 0.5)
ax0.set_xlabel('$k$', fontsize=12)
ax0.set_ylabel('$Re(x[k])$', fontsize=12)

ax1.set_xlim(-0.5, 0.5)
if task == 1 or task == 3:
    ax2.set_xlabel('$\\nu$', fontsize=12)
else: 
    ax1.set_xlabel('$\\nu$', fontsize=12)

ax1.grid(True)
if task == 1:
    ax1.set_ylabel('$|X_N(\\nu)|$', fontsize=12)
if task == 2:
    ax1.set_ylabel('$|X_L(\\nu)|$', fontsize=12)
if task == 3:
    ax1.set_ylabel('$|X_D(\\nu)|$', fontsize=12)
if task == 4:
    ax1.set_ylabel('$|X_S(\\nu)|$', fontsize=12)
if task == 5:
    ax1.set_ylabel('$|X_C(\\nu)|$', fontsize=12)
ax1.grid(visible=True, which='major')
ax1.grid(visible=True, which='minor',c="#DDDDDD")
ax1.tick_params(axis='both', which='major', labelsize=10)
ax1.minorticks_on()

if task == 1 or task == 3:
    ax2.set_xlim(-0.5, 0.5)
    ax2.grid(True)
    ax2.set_ylabel('$arg(X(\\nu))$', fontsize=12)
    ax2.grid(visible=True, which='major')
    ax2.grid(visible=True, which='minor',c="#DDDDDD")
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.minorticks_on()

ax1.plot(np.linspace(-0.5, 0.5, Nd), yabs, c="tab:blue")

if task == 1:
    ax2.plot(np.linspace(-0.5, 0.5, Nd), yph, c="tab:blue")
if task == 3:
    x[0:N] = 1
    y = np.fft.fft(x, Nd)
    y = np.concatenate ((y[int(Nd/2+1):], y[0:int(Nd/2)+1]))
    ax2.plot(np.linspace(-0.5, 0.5, Nd), np.absolute(np.gradient (y) * 1j * Nd / (2 * np.pi)), c="tab:blue")
    ax2.set_ylabel('$|\\frac{j}{2\\pi}\\frac{dX_N}{d\\nu}|$', fontsize=12)
plt.show()