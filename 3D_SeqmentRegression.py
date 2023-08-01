import math
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import random


def lnear(lX, t):
    L = len(lX)
    for i in range(L):
        if t < lX[i]:
            return i-1
        if i == L-1:
            return L-1


def findex(lst, d, eps=0.0001) -> int:
    """_summary_

    あとで二分探索にしたい。

    Args:
        lst (list): list of values. be sorted
        d (float): value
    """
    L = len(lst)
    for i in range(L):
        if abs(lst[i]-d) < eps:
            return i
    return -1


def todense(a1t, count):
    while True:
        if count.count(0) == 0:
            break
        idx = count.index(0)
        if idx == 0:
            breakwall = 1
        else:
            breakwall = idx
        a1t.pop(breakwall)
        if idx == 0:
            a = count.pop(0)
            count[0] += a
        else:
            count[idx-1] += count.pop(idx)
    return a1t, count


def makecount(a1t, d):
    n_t = len(a1t)-1
    N = len(d)
    count = [0 for i in range(n_t)]
    for ip in range(N):
        idx = lnear(a1t, d[ip])
        if idx < 0 or idx > n_t-1:
            continue
        count[idx] += 1
    return count


def makeF(M, A, B, lX):

    def F(x: float):
        idx = lnear(lX, x)
        if idx < 0:
            idx = 0
        elif len(A)-1 < idx:
            idx = len(A)-1
        if np.isnan(A[idx]) or np.isnan(B[idx]):
            return np.nan
        return A[idx]*x+B[idx]
    return F


def polyfit(x, y):
    X = []
    Y = []
    L = x.shape[0]
    for i in range(L):
        if np.isnan(y[i]):
            continue
        else:
            X.append([x[i], 1.0])
            Y.append(y[i])
    if len(X) < 2 or len(Y) < 2:
        return [np.nan, np.nan]
    X = np.array(X)
    Y = np.array(Y)
    # print(X.shape, Y.shape)
    A = np.linalg.solve(X.T@X, X.T@Y)
    # print(A)
    return [A[0], A[1]]


def _3D_SegmentRegression(l_Ts, l_Xs, l_Hs, s_Ts, s_Xs, s_Hs, dT, lam=1.0e3):
    """_summary_

    Args:
        l_Ts (list[float]): クラスタ内の点すべての時刻(t) \n
        l_Xs (list[float]): クラスタ内の点すべての距離(x) \n
        l_Hs (list[float]): クラスタ内の点すべての高度(h) \n
        s_Ts (list[float]): クラスタ内の時刻集合,sorted \n
        s_Xs (list[float]): クラスタ内の距離集合,sorted \n 
        s_Hs (list[float]): クラスタ内の高度集合,sorted \n
        dT (float): 折れ線区間時間幅 \n
        lam (float, optional): 最適化問題の制約項 (Defaults to 1.0e3).
    """
    n_H = len(s_Hs)
    a1h = sorted(s_Hs)
    n_P = len(l_Ts)
    M_x = max(l_Xs)

    colors = []
    for ih in range(n_H):
        colors.append((random(), random(), random()))

    T_div = [[] for ih in range(n_H)]
    X_div = [[] for ih in range(n_H)]
    for ip in range(n_P):
        h = l_Hs[ip]
        hidx = findex(s_Hs, h)
        T_div[hidx].append(l_Ts[ip])
        X_div[hidx].append(l_Xs[ip])

    Ms = []
    As = []
    dAs = []
    Bs = []
    ll_Xs = []
    for ih in range(n_H):
        M, A, B, lX = SegmentRegression(
            T_div[ih], X_div[ih], dT, lam=lam, Tbound=[min(l_Ts)-0.0001, max(l_Ts)+0.0001])
        Ms.append(M)
        As.append(A)
        dA = np.full((M), 0.0, dtype=float)
        for jt in range(1, M, 1):
            dA[jt] = (A[jt]-A[jt-1])/dT
        dAs.append(dA)
        Bs.append(B)
        ll_Xs.append(lX)
    vel_average = []
    for ih in range(n_H):
        vel, dst = polyfit(np.array(T_div[ih]), np.array(X_div[ih]))
        vel_average.append(vel)
    lX = ll_Xs[0]
    M = Ms[0]
    Fs = []
    for i in range(n_H):
        Fs.append(makeF(M, As[i], Bs[i], lX))

    a2t = np.full((len(lX)-1), 0.0, dtype=float)
    for it in range(len(lX)-1):
        a2t[it] = 0.5*(lX[it]+lX[it+1])
    a2xs = np.full((n_H, a2t.shape[0]), np.nan, dtype=float)
    for ih in range(n_H):
        for jt in range(a2t.shape[0]):
            a2xs[ih, jt] = Fs[ih](a2t[jt])
    n_T = 30
    T_ln = np.linspace(min(l_Ts), max(l_Ts), n_T)
    X_ln = np.full((n_T), 0.0, dtype=float)
    for it in range(n_T):
        X_ln[it] = Fs[0](T_ln[it])
    dist = np.full((n_H, n_T), np.nan, dtype=float)
    for ih in range(n_H):
        for jt in range(n_T):
            dist[ih, jt] = Fs[ih](T_ln[jt])
    elev = np.full((n_T), np.nan, dtype=float)
    slopes = np.full((n_T), np.nan, dtype=float)
    base = np.full((n_T), np.nan, dtype=float)
    for jt in range(n_T):
        p, q = polyfit(np.array(a1h), dist[:, jt])
        if not np.isnan(p):
            slopes[jt] = p
            elev[jt] = math.atan(p)*180.0/math.pi
            base[jt] = p*0.0+q

    # 描画
    fig = plt.figure()
    fig.suptitle("Cluster Analysis")
    # クラスタの点群の様子を3Dでプロット
    ax00 = fig.add_subplot(1, 3, 1, projection="3d")
    ax00.scatter(l_Ts, l_Xs, l_Hs, s=5, c="blue")
    ax00.set_xlabel("Time")
    ax00.set_ylabel("Distance")
    ax00.set_zlabel("Height")
    # 各高度・時刻での速度をプロット
    ax01 = fig.add_subplot(2, 3, 2)
    for ih in range(n_H):
        ax01.plot(a2t, As[ih], marker="x", c=colors[ih],
                  label="{i}".format(i=ih))
        ax01.axhline(
            y=vel_average[ih], linestyle="dotted", c=colors[ih], linewidth=3)
    ax01.set_xlabel("Time")
    ax01.set_ylabel("Velocity")
    ax01.legend()
    # 高度・位置ごとの速度をプロット
    ax02 = fig.add_subplot(2, 3, 3)
    for ih in range(n_H):
        ax02.plot(a2xs[ih], As[ih], marker="x",
                  c=colors[ih], label="{i}".format(i=ih))
    ax02.set_xlabel("Distance")
    ax02.set_ylabel("Velocity")
    ax02.legend()
    # 時刻ごとの仰角を調べる(一つの時刻tを固定して、その時刻での傾きを調べる)
    ax11 = fig.add_subplot(2, 3, 5)
    ax11.plot(T_ln, elev)
    ax11.set_xlabel("Time")
    ax11.set_ylabel("Zenith")
    # # 位置ごとの仰角を調べる
    ax12 = fig.add_subplot(2, 3, 6)
    ax12.plot(base, elev)
    ax12.set_xlabel("Distance")
    ax12.set_ylabel("Zenith")

    plt.show()


def SegmentRegression(l_Ts, l_Xs, dT, lam=1.0e3, Tbound=[np.nan, np.nan]):
    """
    クラスタ内の点の座標リストを用いて、適当な区間幅でセグメント線形回帰を行います。\n
    分割した区間内にデータが一つもない場合は適当に周りの区間と結合させる。
    Given :
            l_Ts  : クラスタ内の点の時刻tをリストにしたもの \n
            l_Xs  : クラスタ内の点の距離xをリストにしたもの \n
            dT  : 折れ線を作る時間間隔 \n
            lam : 制約付き最適化におけるハイパーパラメータ \n
            Tbound : 時刻の最端点 \n

    Return : M  int : 分割数 \n
             A  np.ndarray : 各分割における回帰直線の傾き(サイズM) \n
             B  np.ndarray : 各分割における回帰直線の切片(サイズM) \n
             lX np.ndarray : 区間の端点M+1個 \n
    """

    mT = 0.0
    MT = 0.0
    if np.isnan(Tbound[0]):
        mT = min(l_Ts)
    else:
        mT = Tbound[0]
    if np.isnan(Tbound[1]):
        MT = max(l_Ts)
    else:
        MT = Tbound[1]
    N = len(l_Ts)
    M = math.ceil((MT - mT) / dT)
    a1t = []
    for i in range(M+1):
        if mT+dT*i < MT:
            a1t.append(mT+dT*i)
        else:
            if a1t[len(a1t)-1] < MT:
                a1t.append(MT)
    n_t = len(a1t)-1
    count = makecount(a1t, l_Ts)
    COUNT = count.copy()
    a1t, count = todense(a1t, count)
    M = len(a1t)-1

    idx_list = np.full((N), -1, dtype=int)
    XX_seg = np.full((M), 0.0, dtype=float)
    X_seg = np.full((M), 0.0, dtype=float)
    I_seg = np.full((M), 0.0, dtype=float)
    Y_seg = np.full((M), 0.0, dtype=float)
    XY_seg = np.full((M), 0.0, dtype=float)

    for ip in range(N):
        idx = lnear(a1t, l_Ts[ip])
        if idx >= M:
            idx = M - 1
        idx_list[ip] = idx
        XX_seg[idx] += l_Ts[ip] * l_Ts[ip]
        X_seg[idx] += l_Ts[ip]
        I_seg[idx] += 1
        Y_seg[idx] += l_Xs[ip]
        XY_seg[idx] += l_Ts[ip] * l_Xs[ip]

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
    A1T = []
    for i in range(math.ceil((MT - mT) / dT)+1):
        if mT+dT*i < MT:
            A1T.append(mT+dT*i)
        else:
            if A1T[len(A1T)-1] < MT:
                A1T.append(MT)
    AA = np.full((n_t), np.nan, dtype=float)
    BB = np.full((n_t), np.nan, dtype=float)
    idx = 0
    for i in range(n_t):
        if COUNT[i] > 0:
            AA[i] = A[idx]
            BB[i] = B[idx]
            idx += 1
    return n_t, AA, BB, A1T


if __name__ == "__main__":
    n_H = 3
    n_P = 60
    a = np.full((n_H), 0.0, dtype=float)
    b = np.full((n_H), 0.0, dtype=float)
    H = np.linspace(100, 300, 3)
    a[0] = 100.0
    a[1] = 102.0
    a[2] = 105.0
    b[0] = 100.0
    b[1] = 300.0
    b[2] = 500.0
    l_Ts = []
    l_Xs = []
    l_Hs = []
    s_Ts = []
    s_Xs = []
    s_Hs = []
    for i in range(n_H):
        for j in range(n_P):
            Time = 20.0*random()
            Dist = a[i] * Time + b[i] + 3.0 * random()
            Height = H[i]
            l_Ts.append(Time)
            l_Xs.append(Dist)
            l_Hs.append(Height)
            if findex(s_Ts, Time) == -1:
                s_Ts.append(Time)
            if findex(s_Xs, Dist) == -1:
                s_Xs.append(Dist)
            if findex(s_Hs, Height) == -1:
                s_Hs.append(Height)
    # print(l_Ts)
    _3D_SegmentRegression(l_Ts, l_Xs, l_Hs, s_Ts, s_Xs, s_Hs, 1.0, lam=0.0)
    _3D_SegmentRegression(l_Ts, l_Xs, l_Hs, s_Ts, s_Xs, s_Hs, 1.0, lam=1.0e+3)
