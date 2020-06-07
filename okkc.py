# --------------------------------------------------------------------------------------------------------------------
# This python file contains the implementation of OKKC algorithm.
# Reference:
# S. Yu, X. Liu, W. Glanzel, Optimized Data Fusion for Kernel K-means Clustering, IEEE Trans. PAMI, Vol. 34, No. 5,
# 2011.
# Coded by Miao Cheng
# Date: 2020-03-17
# --------------------------------------------------------------------------------------------------------------------
import numpy as np
import scipy.linalg as la
from sklearn import cluster
from scipy import optimize
import sympy
#from scilab import *

from cala import *
from kernel import *


def kms(X, c):
    km = cluster.KMeans(n_clusters=c, random_state=9)
    km.fit_predict(X)
    labels = km.labels_

    return labels


def linsolve_lssvm(omega, L):
    nSam, _ = np.shape(omega)
    nL1, nL2 = np.shape(L)
    
    Y = []
    for i in range(nL2):
        tml = L[:, i]
        td = np.diag(tml)
        Y.append(td)
        
    onevec = np.ones((nL1, 1))
    tmp = []
    tmp.append(0)
    tmp = np.column_stack((tmp, np.transpose(onevec)))
    tmq = np.column_stack((onevec, omega))
    H = np.row_stack((tmp, tmq))
    
    tmp = np.zeros((1, nL2))
    tmq = 1 / L
    J = np.row_stack((tmp, tmq))
    #sol = sympy.linsolve([H], [J])
    #sol = scilab.linsolve(H, J)
    sol = la.solve(H, J)
    
    beta = sol[1::, :]
    b = sol[1, :]
    
    nRow, nCol = np.shape(beta)
    alpha = np.zeros((nRow, nCol))
    for i in range(nL2):
        tmy = Y[i]
        tmb = beta[:, i]
        
        tmp = np.linalg.inv(tmy)
        tmp = np.dot(tmp, tmb)
        
        alpha[:, i] = tmp
        
    return alpha, beta, b


def solve_lp(sis):
    if sis.ndim == 1:
        nLen = len(sis)
        n = 1
        sis = np.reshape(sis, [n, nLen])
    else:
        n, nLen = np.shape(sis)
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # [x,fval]=linprog(c, A, b, Aeq, beq, LB, UB, X0, OPTIONS)
    # LB,UB分别为x的上界和下界
    # 
    # from scipy import optimize
    # import numpy as np
    # 求解函数
    # res = optimize.linprog(c, A, b, Aeq, beq, LB, UB, X0, OPTIONS)
    # 目标函数最小值
    # print(res.fun)
    # 最优解
    # print(res.x)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    c = []
    c.append(-1)
    tmp = np.zeros((1, nLen))
    c = np.column_stack((c, tmp))
    
    tmp = np.ones((n, 1))
    A = np.column_stack((tmp, -sis))
    #A = A[0]
    b = np.zeros((n, 1)).reshape(n, )
    
    tmp = np.ones((1, nLen))
    Aeq = []
    Aeq.append(0)
    Aeq = np.column_stack((Aeq, tmp))
    #Aeq = Aeq[0]
    beq = np.ones((1, 1)).reshape(1, )
    
    tmp = np.zeros((1, nLen))
    LB = []
    LB.append(-1e6)
    LB = np.column_stack((LB, tmp))
    LB = LB[0]
    UB = 1e6 * np.ones((1, nLen+1))
    UB = UB[0]
    
    res = optimize.linprog(c=c, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=None, options=None)
    theta = res.x
    gamma = res.fun
    
    theta = theta[1::]
    
    return theta, gamma
    
    
def calSis(K, L, B):
    nFea = len(K)
    _, c = np.shape(L)
    sis = np.zeros((1, nFea)).reshape(nFea, )
    sb = - np.sum(np.sum(B))
    
    for i in range(nFea-1):
        sis[i] = sb
        tmk = K[i]
        for j in range(c):
            tmb = B[:, j]
            tma = L[:, j]
            da = np.diag(tma)
            
            tmp = np.dot(np.transpose(tmb), da)
            tmp = np.dot(tmp, tmk)
            tmp = np.dot(tmp, da)
            tmp = np.dot(tmp, tmb)
            tmp = 0.5 * tmp
            
            sis[i] = sis[i] + tmp
            
    sis[nFea-1] = sb
    for i in range(c):
        tmb = B[:, i]
        tmk = K[nFea-1]
        tmp = np.dot(np.transpose(tmb), tmk)
        tmp = np.dot(tmp, tmb)
        tmp = 0.5 * tmp
        
        sis[nFea-1] = sis[nFea-1] + tmp
        
        
    return sis
    

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This function defines the multiple class line norm LSSVM MKL solver
# Output:
# Theta: kernel coefficients
# B: The dual variables
# E: The dummy variable checking the convergence
# sis: The dummy variable of f (alpha)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def sip(K, L, nIter):
    nFea = len(K)
    tmk = K[0]
    nSam, _ = np.shape(tmk)
    _, c = np.shape(L)
    
    I = np.eye(nSam)
    K.append(I)
    
    B = np.random.randn(nSam, c) / 2
    sis = calSis(K, L, B)
    
    E = []
    for ii in range(nIter):
        str = 'The %d-th iteration in sip' %ii
        print(str)
        
        theta, gamma = solve_lp(sis)
        omega = np.zeros((nSam, nSam))
        for i in range(nFea):
            tmk = K[i]
            tmp = theta[i] * tmk
            omega = omega + tmp
            
            
        tmp = np.linalg.cond(omega)
        if tmp > 1e18:
            str = 'The matrix is singular, and the initial matrix needs to be regenerated !'
            print(str)
            
            theta = 0
            B = 0
            E = 0
            sis = 0
            return theta, B, E, sis
        
        B, trash2, trash3 = linsolve_lssvm(omega, L)
        S = calSis(K, L, B)
        sis = np.row_stack((sis, S))
        tmp = np.dot(S, theta)
        tmp = tmp / gamma
        eps = 1 + tmp
        
        E.append(eps)
        if abs(eps) < 5e-5:
            break
            
            
    return theta, B, E, sis
    
    
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This function defines the transformation for generation of binary cluster 21
# assignment matrix.
# Z - nSam * c the binary cluster assignment matrix
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def l2p(labels):
    nSam = len(labels)
    nL = np.max(labels)
    
    Z = np.zeros((nSam, nL+1))
    Z[:, labels] = 1
    
    # +++++ Ensure there is no null assignment +++++
    for i in range(nL+1):
        tmp = Z[:, i]
        tmp = np.sum(tmp, axis=0)
        if tmp == 0:
            ind = np.array(range(nL+1))
            np.random.shuffle(ind)
            ind = ind[0]
            Z[ind, :] = np.zeros((1, nL+1))
            Z[ind, i] = 1
    
    return Z
    

def okkc(Fea, c, nIter):
    nFea = len(Fea)
    tmx = Fea[0]
    nDim, nSam = np.shape(tmx)
    # +++++ Initialize centered matrix +++++
    e = np.ones((nSam, 1))
    tmp = np.dot(e, np.transpose(e))
    tmp = (float(1) / nSam) * tmp
    I = np.eye(nSam)
    P = I - tmp
    
    K = []
    mk = np.zeros((nSam, nSam))
    for i in range(nFea):
        tmx = Fea[i]
        tmk = Kernel(tmx, tmx, 'Gaussian')
        
        tnk = np.dot(P, tmk)
        tnk = np.dot(tnk, P)
        tnk = tnk + np.transpose(tnk)
        tnk = tnk / 2
        K.append(tnk)
        mk = mk + tmk
        
    mk = mk / nFea
    theta = np.ones((nFea, 1)).reshape(nFea, )
    theta = theta / nFea
    
    #tol = 0.1
    U, s, V = la.svd(mk, full_matrices=False)
    V = U[:, 0:c]
    
    # +++++ Perform standard K-Means Clustering +++++
    labels = kms(V, c)
    
    Z = l2p(labels)
    
    obj_history = []
    #old_obj = -1e7
    tol = 0.1
    for ii in range(nIter):
        L = 2 * Z
        L = L - 1
        
        # +++++ Initialize parameters +++++
        mu = 0
        B = 0
        E = 0
        sis = 0
        
        # +++++ Optimized by LSSVM Linf MKL +++++
        mu, B, E, sis = sip(K, L, nIter)
        
        tmk = K[i]
        nRow, nCol = np.shape(tmk)
        Koptimal = np.zeros((nRow, nCol))
        for i in range(nFea):
            tmk = K[i]
            tmp = tmk * mu[i]
            Koptimal = Koptimal + tmp
            
            I = np.eye(nSam)
            tmp = mu[nFea-1] * I
            Koptimal = Koptimal + tmp
            tmp = np.dot(np.transpose(P), Koptimal)
            Koptimal = np.dot(tmp, P)
            
            tmp = np.dot(Koptimal, Koptimal)
            
            # +++++ Check the singularity +++++
            #temp = np.linalg.det(tmp)
            #if np.abs(temp) < 1e-6:
                #tmp = tmp + 1e-6 * np.eye(nSam)
            
            #U, s, _ = la.svd(tmp)
            s, U = la.eig(tmp)
            s = np.real(s)
            U = np.real(U)
            ind = np.argsort(- s)
            s = s[ind]
            U = U[:, ind]
            
            Vnew = U[:, 0:c]
            
            labels = kms(Vnew, c)
            Znew = l2p(labels)
            thetanew = np.transpose(mu)
            
            tmp = V - Vnew
            tmp = norm(tmp, 2)
            tmq = norm(Vnew, 2)
            obj = tmp / tmq
            obj_history.append(obj)
            
            str = 'The %d-th iteration: ' %ii + '%f\n' %obj
            print(str)
            
            if len(obj_history) > 4:
                if obj < tol:
                    break
                
            Z = Znew
            theta = thetanew
            V = Vnew
            
            
    labels = labels + 1
    
    
    return labels, Z, theta
