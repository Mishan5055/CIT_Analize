import math
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


def _3D_SegmentRegression(Ts, Xs, Hs, dT, lam=1.0e3):
    n_H = max(Hs) + 1
    n_P = len(Ts)

    T_div = [[] for ih in range(n_H)]
    X_div = [[] for ih in range(n_H)]
    for ip in range(n_P):
        h = Hs[ip]
        T_div[h].append(Ts[ip])
        X_div[h].append(Xs[ip])

    Ms = []
    As = []
    Bs = []
    lXs = []
    for ih in range(n_H):
        M, A, B, lX = SegmentRegression(T_div[ih], X_div[ih], dT, lam)
        print(ih, A, B)
        Ms.append(M)
        As.append(A)
        Bs.append(B)
        lXs.append(lX)


def SegmentRegression(Ts, Xs, dT, lam=1.0e3):
    N = len(Ts)
    mT = min(Ts)
    MT = max(Ts)
    M = math.ceil((MT - mT) / dT)

    idx_list = np.full((N), -1, dtype=int)
    XX_seg = np.full((M), 0.0, dtype=float)
    X_seg = np.full((M), 0.0, dtype=float)
    I_seg = np.full((M), 0.0, dtype=float)
    Y_seg = np.full((M), 0.0, dtype=float)
    XY_seg = np.full((M), 0.0, dtype=float)

    for ip in range(N):
        idx = math.floor((Ts[ip] - mT) / dT)
        if idx >= M:
            idx = M - 1
        idx_list[ip] = idx
        XX_seg[idx] += Ts[ip] * Ts[ip]
        X_seg[idx] += Ts[ip]
        I_seg[idx] += 1
        Y_seg[idx] += Xs[ip]
        XY_seg[idx] += Ts[ip] * Xs[ip]

    K = lil_matrix((2 * M, 2 * M))
    for i in range(M):
        K[i, i] = XX_seg[i]
        K[i + M, i] = X_seg[i]
        K[i, i + M] = X_seg[i]
        K[i + M, i + M] = I_seg[i]
    L = np.full((2 * M, 1), 0.0, dtype=float)
    for i in range(M):
        L[i, 0] = XY_seg[i]
        L[i + M, 0] = Y_seg[i]

    _M = lil_matrix((M - 1, 2 * M))
    for i in range(M - 1):
        _M[i, i] = dT * (i + 1)
        _M[i, i + 1] = -dT * (i + 1)
        _M[i, i + M] = 1.0
        _M[i, i + M + 1] = -1.0

    csr_K = csr_matrix(K)
    csr_L = csr_matrix(L)
    csr_M = csr_matrix(_M)

    P = spsolve(csr_K.T * csr_K + lam * lam * csr_M.T * csr_M, csr_K.T * csr_L)
    A = P[:M]
    B = P[M:]
    X = np.linspace(0.0, dT * M, M + 1)
    return M, A, B, X


if __name__ == "__main__":
    n_H = 3
    n_P = 20
    a = np.full((n_H), 0.0, dtype=float)
    b = np.full((n_H), 0.0, dtype=float)
    a[0] = 1.0
    a[1] = 2.0
    a[2] = 3.0
    b[0] = 1.0
    b[1] = 2.0
    b[2] = 3.0
    T = np.linspace(0.0, 3.0, n_P)
    Ts = []
    Xs = []
    Hs = []
    for i in range(n_H):
        for j in range(n_P):
            Ts.append(T[j])
            Xs.append(a[i] * T[j] + b[i])
            Hs.append(i)
    _3D_SegmentRegression(Ts, Xs, Hs, 0.5)
