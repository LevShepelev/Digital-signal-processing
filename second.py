import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

x = np.asarray ([1, -6, 0, -4, 6, -7, 0, 9])
m = 4

X1 = np.fft.fft (x)

Wtr = np.matrix(sc.linalg.dft(x.shape[0]))
iWtr = Wtr.getH()

X2 = Wtr @ x

print ('Spectrum difference = ', sc.linalg.norm (X1 - X2))

fig, axs = plt.subplots(2, 2, sharey=False)
fig.tight_layout (pad=2.0)

for i in range(0, 2):
    for j in range(0, 2):
        axs[i][j].grid(True)
        axs[i][j].grid(visible=True, which='major')
        axs[i][j].grid(visible=True, which='minor',c="#DDDDDD")
        axs[i][j].tick_params(axis='both', which='major', labelsize=10)
        axs[i][j].minorticks_on()
        axs[i][j].set_xlim(-0.1, X1.shape[0]-0.9)
        axs[i][j].set_xlabel('$k$', fontsize=12)
        axs[i][j].set_ylim(-np.max (np.absolute(X1)) - 0.5, np.max (np.absolute(X1)) + 0.5)
    
axs[0][0].set_ylabel('$Re(X[k])$', fontsize=12)
axs[0][0].stem(range (0, X1.shape[0]), np.real(X1))

axs[0][1].set_ylabel('$Im(X[k])$', fontsize=12)
axs[0][1].stem(range (0, X1.shape[0]), np.imag(X1))

axs[1][0].set_ylabel('$|X[k]|$', fontsize=12)
axs[1][0].stem(range (0, X1.shape[0]), np.absolute(X1))

axs[1][1].set_ylabel('$arg(X[k])$', fontsize=12)
axs[1][1].stem(range (0, X1.shape[0]), np.angle(X1))
axs[1][1].set_ylim(-np.max (np.angle(X1)) - 0.5, np.max (np.angle(X1)) + 0.5)

plt.show()

y = np.zeros(X1.shape[0])
y = np.fft.ifft (X1 * np.exp (1j * 2 * np.pi * np.asarray(range (0, x.shape[0])) / x.shape[0] * m))

print ('x[k] = ', x)
print ('y[k] = ', np.rint(np.real(y)).astype(int))
print ('||Im(y)|| = ', sc.linalg.norm(np.imag(y)))

fig, axs = plt.subplots(2, 1)
fig.tight_layout (pad=2.0)

for i in range(0, 2):
    axs[i].grid(True)
    axs[i].grid(visible=True, which='major')
    axs[i].grid(visible=True, which='minor',c="#DDDDDD")
    axs[i].tick_params(axis='both', which='major', labelsize=10)
    axs[i].minorticks_on()
    axs[i].set_xlim(-0.1, x.shape[0]-0.9)
    axs[i].set_xlabel('$k$', fontsize=12)
    axs[i].set_ylim(-np.max (np.absolute(x)) - 0.5, np.max (np.absolute(x)) + 0.5)
    
axs[0].set_ylabel('$Re(x[k])$', fontsize=12)
axs[0].stem(range (0, x.shape[0]), x)

axs[1].set_ylabel('$Re(y[k])$', fontsize=12)
axs[1].stem(range (0, x.shape[0]), np.real(y))

plt.show()