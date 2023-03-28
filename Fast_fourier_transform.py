import time
import matplotlib.pyplot as plt
import numpy as np

#This is custom version of Fast Fourier Transform, it works only with 2 ^ N points

def dtft(posx, negx, nu):
    sum = 0
    if posx is not None:
        for i in range (posx.shape[0]):
            sum += posx[i] * np.exp (-2j * np.pi * nu * i)

    if negx is not None:
        for i in range (negx.shape[0]):
            sum += negx[i] * np.exp (2j * np.pi * nu * (i+1))

    return sum


def number_reverse (num : np.uint, nbase : np.uint):
    result : np.uint = 0
    for i in range(nbase):
        result <<= 1
        result |= num & 1
        num >>= 1
    return result


def array_reverse (arr, nbase : np.uint):
    for x in range (0, arr.size):
        xrev = number_reverse (x, nbase)
        if (xrev > x):
            arr[x], arr[xrev] = arr[xrev], arr[x]


def myfft (arr, nbase : np.uint):
    arrt = np.array(arr[:], dtype=complex)
    array_reverse(arrt, nbase)
    fourier_coeff = np.exp (-2j * np.pi * np.arange (0, 2 ** (nbase - 1), dtype=complex) / (2 ** nbase))
    for i in range(nbase): # iteration  in distance for two points fft
        r = 2 ** i 
        nr = 2 ** (nbase - i) 
        for j in range (nr//2): # iteration between groups
            for k in range (r): # iteration in one group
                curpos = 2 * r * j + k
                arrt[curpos] = arrt[curpos] + arrt[curpos + r]
                arrt[curpos + r] = arrt[curpos] - 2*arrt[curpos+r]
                if j % 2 == 1:
                    arrt[curpos] *= fourier_coeff[k * (nr//4)]
                    arrt[curpos + r] *= fourier_coeff[(k + r) * (nr//4)]
    return arrt

r = 14
Number_of_points = 2 ** r
xarr = np.arange (0, Number_of_points, 1) + 1
tm0 = time.time ()
for i in range (3):
    np.fft.fft (xarr)

tm1 = time.time()
for i in range (3):
    myfft (xarr, r)

tm2 = time.time()
for i in range (3):
    dtft (xarr, None, np.linspace (0, 1, Number_of_points + 1))

tm3 = time.time()
print (np.max(np.absolute(np.fft.fft (xarr) - myfft (xarr, r))))
print ("Number_of_points = ", Number_of_points, "library version = ", tm1 - tm0, ' ', tm2 - tm1, ' ', tm3 - tm2)
