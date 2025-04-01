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
    fig.suptitle("Solution, Effective Freq: " + str(L) + "," + str(i) + " of " + str(num))
    ax2.contour(resmesh)
    plt.show()

def main():
    # import
    mu = 0.80
    data = (1/(h**4))*np.load('080_diag000.npy')
    # solve v1
    result = np.linalg.eig(data)
    eigenvalues = np.real(result[0])
    eigenvectors = np.real(result[1])
    print(len(eigenvalues))
    order = np.argsort(eigenvalues)
    # find effective frequency
    eff_freq = np.zeros_like(eigenvalues)
    firstval = True
    start = 0
    for i in range(0,2704):
        ind = order[i]
        if eigenvalues[ind] > 0:
            eff_freq[ind] = np.sqrt(eigenvalues[ind]*(1-mu**2))
            if firstval:
                print("set to",i)
                start = i
                firstval = False
    # plotting eigenstates
    RANGE = 20
    for i in range(start,start+RANGE):
        index = order[i]
        plot(eigenvectors[:,index],eff_freq[index],2704,index)
    # plotting effective frequency
    for i in range(start,start+RANGE):
        index = order[i]
        plt.scatter(i,eff_freq[index])
    plt.title(r'$\mu=$'+str(mu))
    plt.ylabel('Effective freq.')
    plt.xlabel('Order')
    plt.show()

if __name__ == '__main__':
    main()