# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 23:10:51 2025

@author: bjama
"""

import numpy as np
import matplotlib.pyplot as plt


N = 52
h = 0.01

def vectomesh(vector,N):
    mesh = np.zeros((N,N))
    for nx in range(0,N):
        for ny in range(0,N):
            mesh[nx,ny] = vector[(nx*N)+ny]
    return mesh

def plot(vector,L,num=1,i=1):
    resmesh = vectomesh(vector,N)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    cax = ax1.pcolormesh(resmesh)
    fig.colorbar(cax,ax=ax1,orientation='vertical')
    fig.suptitle("Solution, Eigenvalue: " + str(L) + "," + str(i) + " of " + str(num))
    ax2.contour(resmesh)
    plt.show()

def main():
    # import
    data = (1/(h**4))*np.load('newmatrix030.npy')
    # solve v1
    result = np.linalg.eig(data)
    eigenvalues = np.real(result[0])
    eigenvectors = np.real(result[1])
    print(len(eigenvalues))
    for i in range(0,10):
        plot(eigenvectors[:,i],eigenvalues[i],2704,i)

if __name__ == '__main__':
    main()