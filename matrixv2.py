# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 18:51:49 2025

@author: bjama
"""

import numpy as np

MU = 0.3
N = 52

def index(nx,ny):
    # gives vector index for nx,ny
    temp = (nx*(N)) + ny
    return temp

def meshtovec(mesh):
    vector = np.zeros(N**2)
    # converts a given spatial matrix into vector form
    for nx in range(0,N):
        for ny in range(0,N):
            vector[int(index(nx,ny))] = mesh[nx,ny]
    return vector

# 1st ghost points

def gl1_central(nx,ny,c):
    # equivalent of a central ghost point on left surface
    xs = np.array([nx+1,nx+1,nx+1,nx+2])
    ys = np.array([ny+1,ny,ny-1,ny])
    cs = c*np.array([-MU,2-2*MU,-MU,-1])
    return xs,ys,cs

def gl1_up(nx,ny,c):
    # up-sided ghost point on left surface
    xs = np.array([nx+1,nx+1,nx+1,nx+1,nx+2])
    ys = np.array([ny+3,ny+2,ny+1,ny,ny])
    cs = c*np.array([MU,-4*MU,5*MU,2-2*MU,-1])
    return xs,ys,cs

def gl1_down(nx,ny,c):
    # down-sided ghost point on left surface
    xs = np.array([nx+1,nx+1,nx+1,nx+1,nx+2])
    ys = np.array([ny,ny-1,ny-2,ny-3,ny])
    cs = c*np.array([2-2*MU,5*MU,-4*MU,MU,-1])
    return xs,ys,cs

def gr1_central(nx,ny,c):
    # equivalent of a central ghost point on right surface
    xs = np.array([nx-2,nx-1,nx-1,nx-1])
    ys = np.array([ny,ny+1,ny,ny-1])
    cs = c*np.array([-1,-MU,2-2*MU,-MU])
    return xs,ys,cs

def gr1_up(nx,ny,c):
    # up-sided ghost point on right surface
    xs = np.array([nx-2,nx-1,nx-1,nx-1,nx-1])
    ys = np.array([ny,ny+3,ny+2,ny+1,ny])
    cs = c*np.array([-1,MU,-4*MU,5*MU,2-2*MU])
    return xs,ys,cs

def gr1_down(nx,ny,c):
    # down-sided ghost point on right surface
    xs = np.array([nx-2,nx-1,nx-1,nx-1,nx-1])
    ys = np.array([ny,ny,ny-1,ny-2,ny-3])
    cs = c*np.array([-1,2-2*MU,5*MU,-4*MU,MU])
    return xs,ys,cs

def gt1_central(nx,ny,c):
    # equivalent of a central ghost point on top surface
    xs = np.array([nx-1,nx,nx,nx+1])
    ys = np.array([ny-1,ny-1,ny-2,ny-1])
    cs = c*np.array([-MU,2-2*MU,-1,-MU])
    return xs,ys,cs

def gt1_left(nx,ny,c):
    # left-sided ghost point on top surface
    xs = np.array([nx-3,nx-2,nx-1,nx,nx])
    ys = np.array([ny-1,ny-1,ny-1,ny-1,ny-2])
    cs = c*np.array([MU,-4*MU,5*MU,2-2*MU,-1])
    return xs,ys,cs

def gt1_right(nx,ny,c):
    # right-sided ghost point on top surface
    xs = np.array([nx,nx,nx+1,nx+2,nx+3])
    ys = np.array([ny-1,ny-2,ny-1,ny-1,ny-1])
    cs = c*np.array([2-2*MU,-1,5*MU,-4*MU,MU])
    return xs,ys,cs

def gb1_central(nx,ny,c):
    # equivalent of a central ghost point on bottom surface
    xs = np.array([nx-1,nx,nx,nx+1])
    ys = np.array([ny+1,ny+2,ny+1,ny+1])
    cs = c*np.array([-MU,-1,2-2*MU,-MU])
    return xs,ys,cs

def gb1_left(nx,ny,c):
    # left-sided ghost point on bottom surface
    xs = np.array([nx-3,nx-2,nx-1,nx,nx])
    ys = np.array([ny+1,ny+1,ny+1,ny+1,ny+2])
    cs = c*np.array([MU,-4*MU,5*MU,2-2*MU,-1])
    return xs,ys,cs

def gb1_right(nx,ny,c):
    # right-sided ghost point on bottom surface
    xs = np.array([nx,nx,nx+1,nx+2,nx+3])
    ys = np.array([ny+2,ny+1,ny+1,ny+1,ny+1])
    cs = c*np.array([-1,2-2*MU,5*MU,-4*MU,MU])
    return xs,ys,cs

# 2nd ghost points

def gl2(nx,ny,c):
    # 2nd ghost point on left surface
    xs = np.array([nx+1,nx+1,nx+1,nx+3,nx+3,nx+3,nx+4])
    ys = np.array([ny+1,ny,ny-1,ny+1,ny,ny-1,ny])
    cs = c*np.array([MU-2,6-2*MU,MU-2,2-MU,2*MU-6,2-MU,1])
    return xs,ys,cs

def gr2(nx,ny,c):
    # 2nd ghost point on right surface
    xs = np.array([nx-1,nx-1,nx-1,nx-3,nx-3,nx-3,nx-4])
    ys = np.array([ny+1,ny,ny-1,ny+1,ny,ny-1,ny])
    cs = c*np.array([MU-2,6-2*MU,MU-2,2-MU,2*MU-6,2-MU,1])
    return xs,ys,cs

def gt2(nx,ny,c):
    # 2nd ghost point on top surface
    xs = np.array([nx-1,nx-1,nx,nx,nx,nx+1,nx+1])
    ys = np.array([ny-1,ny-3,ny-1,ny-3,ny-4,ny-1,ny-3])
    cs = c*np.array([MU-2,2-MU,6-2*MU,2*MU-6,1,MU-2,2-MU])
    return xs,ys,cs

def gb2(nx,ny,c):
    # 2nd ghost point on bottom surface
    xs = np.array([nx-1,nx-1,nx,nx,nx,nx+1,nx+1])
    ys = np.array([ny+3,ny+1,ny+4,ny+3,ny+1,ny+3,ny+1])
    cs = c*np.array([2-MU,MU-2,1,2*MU-6,6-2*MU,2-MU,MU-2])
    return xs,ys,cs

# corner 1 schemes

def genrow(nx,ny):
    # generates a row of the final matrix
    row = np.zeros((N,N))
    # let's say that (x,y) = (n,m) = row[n,m]
    # generate the relevant coordinates
    # and their coefficients
    xinds = np.array([nx-2,nx-1,nx-1,nx-1,
                      nx,nx,nx,nx,nx,nx+1,nx+1,nx+1,nx+2],dtype=int)
    yinds = np.array([ny,ny+1,ny,ny-1,ny+2,ny+1,ny,ny-1,
                      ny-2,ny+1,ny,ny-1,ny],dtype=int)
    cvals = np.array([1,2,-8,2,1,-8,20,-8,1,2,-8,2,1])
    # we will continuously iterate through this list.
    # where we have ghost points, evaluate until none left
    # remove points evaluated
    finished = False
    while not finished:
        xtemps = np.array([])
        ytemps = np.array([])
        ctemps = np.array([])
        for i in range(0,len(xinds)):
            n = int(xinds[i])
            m = int(yinds[i])
            c = cvals[i]
            # 0 value means 0 value
            if cvals[i] == 0:
                continue
            # check for normal points
            if 0<=n<N and 0<=m<N:
                row[n,m] += c
                continue
            # check for second ghost point (left)
            if n == -2:
                res = gl2(n,m,c)
                xtemps = np.append(xtemps,res[0])
                ytemps = np.append(ytemps,res[1])
                ctemps = np.append(ctemps,res[2])
                continue
            # check for second ghost point (right)
            if n == N+1:
                res = gr2(n,m,c)
                xtemps = np.append(xtemps,res[0])
                ytemps = np.append(ytemps,res[1])
                ctemps = np.append(ctemps,res[2])
                continue
            # second ghost point (bottom)
            if m == -2:
                res = gb2(n,m,c)
                xtemps = np.append(xtemps,res[0])
                ytemps = np.append(ytemps,res[1])
                ctemps = np.append(ctemps,res[2])
                continue
            # second ghost point (top)
            if m == N+1:
                res = gt2(n,m,c)
                xtemps = np.append(xtemps,res[0])
                ytemps = np.append(ytemps,res[1])
                ctemps = np.append(ctemps,res[2])
                continue
            # first ghost point (left side)
            if n == -1:
                # central
                if 0<m<N-1:
                    res = gl1_central(n,m,c)
                    xtemps = np.append(xtemps,res[0])
                    ytemps = np.append(ytemps,res[1])
                    ctemps = np.append(ctemps,res[2])
                    continue
                # upwards
                if m == 0:
                    res = gl1_up(n,m,c)
                    xtemps = np.append(xtemps,res[0])
                    ytemps = np.append(ytemps,res[1])
                    ctemps = np.append(ctemps,res[2])
                    continue
                # downwards
                if m == N-1:
                    res = gl1_down(n,m,c)
                    xtemps = np.append(xtemps,res[0])
                    ytemps = np.append(ytemps,res[1])
                    ctemps = np.append(ctemps,res[2])
                    continue
                # corner point 1
                if m == -1:
                    res = gl1_up(n,m,c/2)
                    xtemps = np.append(xtemps,res[0])
                    ytemps = np.append(ytemps,res[1])
                    ctemps = np.append(ctemps,res[2])
                    # no continue. for other term
                # corner point 2
                if m == N:
                    res = gl1_down(n,m,c/2)
                    xtemps = np.append(xtemps,res[0])
                    ytemps = np.append(ytemps,res[1])
                    ctemps = np.append(ctemps,res[2])
                    # again, no continue
            # first ghost point (right side)
            if n == N:
                # central
                if 0<m<N-1:
                    res = gr1_central(n,m,c)
                    xtemps = np.append(xtemps,res[0])
                    ytemps = np.append(ytemps,res[1])
                    ctemps = np.append(ctemps,res[2])
                    continue
                # upwards
                if m == 0:
                    res = gr1_up(n,m,c)
                    xtemps = np.append(xtemps,res[0])
                    ytemps = np.append(ytemps,res[1])
                    ctemps = np.append(ctemps,res[2])
                    continue
                # downwards
                if m == N-1:
                    res = gr1_down(n,m,c)
                    xtemps = np.append(xtemps,res[0])
                    ytemps = np.append(ytemps,res[1])
                    ctemps = np.append(ctemps,res[2])
                    continue
                # corner point 1
                if m == -1:
                    res = gr1_up(n,m,c/2)
                    xtemps = np.append(xtemps,res[0])
                    ytemps = np.append(ytemps,res[1])
                    ctemps = np.append(ctemps,res[2])
                    # no continue. for other term
                # corner point 2
                if m == N:
                    res = gr1_down(n,m,c/2)
                    xtemps = np.append(xtemps,res[0])
                    ytemps = np.append(ytemps,res[1])
                    ctemps = np.append(ctemps,res[2])
                    # again, no continue
            # first ghost point (bottom)
            if m == -1:
                # central
                if 0<n<N-1:
                    res = gb1_central(n,m,c)
                    xtemps = np.append(xtemps,res[0])
                    ytemps = np.append(ytemps,res[1])
                    ctemps = np.append(ctemps,res[2])
                    continue
                # rightwards
                if n == 0:
                    res = gb1_right(n,m,c)
                    xtemps = np.append(xtemps,res[0])
                    ytemps = np.append(ytemps,res[1])
                    ctemps = np.append(ctemps,res[2])
                    continue
                # leftwards
                if n == N-1:
                    res = gb1_left(n,m,c)
                    xtemps = np.append(xtemps,res[0])
                    ytemps = np.append(ytemps,res[1])
                    ctemps = np.append(ctemps,res[2])
                    continue
                # corner point 1
                if n == -1:
                    res = gb1_right(n,m,c/2)
                    xtemps = np.append(xtemps,res[0])
                    ytemps = np.append(ytemps,res[1])
                    ctemps = np.append(ctemps,res[2])
                    # no continue
                if n == N:
                    res = gb1_left(n,m,c/2)
                    xtemps = np.append(xtemps,res[0])
                    ytemps = np.append(ytemps,res[1])
                    ctemps = np.append(ctemps,res[2])
                    # no continue
            # first ghost point (top)
            if m == N:
                # central
                if 0<n<N-1:
                    res = gt1_central(n,m,c)
                    xtemps = np.append(xtemps,res[0])
                    ytemps = np.append(ytemps,res[1])
                    ctemps = np.append(ctemps,res[2])
                    continue
                # rightwards
                if n == 0:
                    res = gt1_right(n,m,c)
                    xtemps = np.append(xtemps,res[0])
                    ytemps = np.append(ytemps,res[1])
                    ctemps = np.append(ctemps,res[2])
                    continue
                # leftwards
                if n == N-1:
                    res = gt1_left(n,m,c)
                    xtemps = np.append(xtemps,res[0])
                    ytemps = np.append(ytemps,res[1])
                    ctemps = np.append(ctemps,res[2])
                    continue
                # corner point 1
                if n == -1:
                    res = gt1_right(n,m,c/2)
                    xtemps = np.append(xtemps,res[0])
                    ytemps = np.append(ytemps,res[1])
                    ctemps = np.append(ctemps,res[2])
                    # no continue
                # corner point 2
                if n == N:
                    res = gt1_left(n,m,c/2)
                    xtemps = np.append(xtemps,res[0])
                    ytemps = np.append(ytemps,res[1])
                    ctemps = np.append(ctemps,res[2])
                    # no continue
        # update the arrays to the new unassigned points
        xinds = xtemps
        yinds = ytemps
        cvals = ctemps
        if xinds.size == 0:
            finished = True
    return row

def genmatrix():
    matrix = np.zeros((N**2,N**2))
    for nx in range(0,N):
        for ny in range(0,N):
            result = genrow(nx,ny)
            matrix[int(index(nx,ny)),:] = meshtovec(result)
    return matrix

def main():
    mat = genmatrix()
    np.save('newmatrix030',mat)

if __name__ == '__main__':
    main()