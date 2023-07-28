import math
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import random

def lnear(lX,t):
    L=len(lX)
    for i in range(L):
        if t<lX[i]:
            return i-1
        if i==L-1:
            return L-1




def makeF(M, A, B, lX):

    def F(x :float):
        idx = lnear(lX,x)
        if idx < 0:
            idx = 0
        elif len(A)-1 < idx:
            idx = len(A)-1

        return A[idx]*x+B[idx]
    return F


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
    dAs = []
    Bs = []
    lXs = []
    for ih in range(n_H):
        M, A, B, lX = SegmentRegression(T_div[ih], X_div[ih], dT, lam=lam,Tbound=[min(Ts),max(Ts)])
        print(ih, M, A, B, lX)
        Ms.append(M)
        As.append(A)
        dA=np.full((M),0.0,dtype=float)
        for jt in range(1,M,1):
            dA[jt]=(A[jt]-A[jt-1])/dT
        dAs.append(dA)
        Bs.append(B)
        lXs.append(lX)
    lX = lXs[0]
    M = Ms[0]
    Fs = []
    for i in range(n_H):
        Fs.append(makeF(M,As[i],Bs[i],lX))

    a2t = np.full((len(lX)-1),0.0,dtype=float)
    for it in range(len(lX)-1):
        a2t[it]=0.5*(lX[it]+lX[it+1])
    a2x = np.full((len(lX)-1),0.0,dtype=float)
    for it in range(len(lX)-1):
        a2x[it]=Fs[0](a2t[it])

    n_T = 300
    T_ln = np.linspace(min(Ts),max(Ts),n_T)
    X_ln = np.full((n_T),0.0,dtype=float)
    for it in range(n_T):
        X_ln[it]=Fs[0](T_ln[it])
    dist_for_elevation = np.full((n_H,n_T),0.0,dtype=float)
    for ih in range(n_H):
        for jt in range(n_T):
            dist_for_elevation[ih,jt]=Fs[ih](T_ln[jt])
    elev=np.full((n_T),0.0,dtype=float)
    slopes=np.full((n_T),0.0,dtype=float)
    for jt in range(n_T):
        p = np.polyfit(np.arange(0,n_H,1),dist_for_elevation[:,jt],1)
        slopes[jt]=p[0]
        elev[jt]=math.atan(p[0])*180.0/math.pi

    # 描画
    fig = plt.figure()
    fig.suptitle("Cluster Analysis")
    # クラスタの点群の様子を3Dでプロット
    ax00 = fig.add_subplot(1,3,1,projection="3d")
    ax00.scatter(Ts,Xs,Hs,s=5,c="blue")
    ax00.set_xlabel("Time")
    ax00.set_ylabel("Distance")
    ax00.set_zlabel("Height")
    # 各高度・時刻での速度をプロット
    ax01 = fig.add_subplot(2,3,2)
    for ih in range(n_H):
        ax01.plot(a2t,As[ih],label="{i}".format(i=ih))
    ax01.set_xlabel("Time")
    ax01.set_ylabel("Velocity")
    ax01.legend()
    # 高度・位置ごとの速度をプロット
    ax02 = fig.add_subplot(2,3,3)
    for ih in range(n_H):
        ax02.plot(a2x,As[ih],label="{i}".format(i=ih))
    ax02.set_xlabel("Distance")
    ax02.set_ylabel("Velocity")
    ax02.legend()
    # 時刻ごとの仰角を調べる(一つの時刻tを固定して、その時刻での傾きを調べる)
    ax11 = fig.add_subplot(2,3,5)
    ax11.plot(T_ln,elev)
    ax11.set_xlabel("Time")
    ax11.set_ylabel("Zenith")
    # 位置ごとの仰角を調べる
    ax12 = fig.add_subplot(2,3,6)
    ax12.plot(X_ln,elev)
    ax12.set_xlabel("Distance")
    ax12.set_ylabel("Zenith")

    plt.show()




def SegmentRegression(Ts, Xs, dT, lam=1.0e3,Tbound=[np.nan,np.nan]):
    """
    クラスタ内の点の座標リストを用いて、適当な区間幅でセグメント線形回帰を行います。\n
    Given :
            Ts  : クラスタ内の点の時刻tをリストにしたもの \n
            Xs  : クラスタ内の点の距離xをリストにしたもの \n
            dT  : 折れ線を作る時間間隔 \n
            lam : 制約付き最適化におけるハイパーパラメータ \n
            Tbound : 時刻の最端点 \n

    Return : M  int : 分割数 \n
             A  np.ndarray : 各分割における回帰直線の傾き(サイズM) \n
             B  np.ndarray : 各分割における回帰直線の切片(サイズM) \n
             lX np.ndarray : 区間の端点M+1個 \n
    """

    mT=0.0
    MT=0.0
    if Tbound[0]==np.nan:
        mT=min(Ts)
    else:
        mT=Tbound[0]
    if Tbound[1]==np.nan:
        MT=max(Ts)
    else:
        MT=Tbound[1]

    N = len(Ts)
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
    X = np.linspace(mT, mT + dT * M, M + 1)
    return M, A, B, X


if __name__ == "__main__":
    n_H = 3
    n_P = 20
    a = np.full((n_H), 0.0, dtype=float)
    b = np.full((n_H), 0.0, dtype=float)
    a[0] = 1.0
    a[1] = 2.0
    a[2] = 4.0
    b[0] = 1.0
    b[1] = 2.0
    b[2] = 3.0
    T = np.linspace(1.0, 4.0, n_P)
    Ts = []
    Xs = []
    Hs = []
    for i in range(n_H):
        for j in range(n_P):
            Ts.append(T[j])
            Xs.append(a[i] * T[j] + b[i] + 0.4 * random())
            Hs.append(i)
    _3D_SegmentRegression(Ts, Xs, Hs, 0.5)
