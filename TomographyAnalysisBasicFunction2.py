import numpy as np
from TomographyAnalysisBasicFunction import SETTING, EXPERIMENT
from TomographyAnalysisBasicFunction import (
    lnearst,
    ImportDTomoSequence,
    L2_on_ellipsoid,
)

from scipy.stats import zscore
from scipy.cluster.hierarchy import linkage, fcluster
import os
import matplotlib.pyplot as plt
from math import ceil
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
            return i - 1
        if i == L - 1:
            return L - 1


def makeF(M, A, B, lX):
    def F(x: float):
        idx = lnear(lX, x)
        if idx < 0:
            idx = 0
        elif len(A) - 1 < idx:
            idx = len(A) - 1

        return A[idx] * x + B[idx]

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
        M, A, B, lX = SegmentRegression(
            T_div[ih], X_div[ih], dT, lam=lam, Tbound=[min(Ts), max(Ts)]
        )
        print(ih, M, A, B, lX)
        Ms.append(M)
        As.append(A)
        dA = np.full((M), 0.0, dtype=float)
        for jt in range(1, M, 1):
            dA[jt] = (A[jt] - A[jt - 1]) / dT
        dAs.append(dA)
        Bs.append(B)
        lXs.append(lX)
    lX = lXs[0]
    M = Ms[0]
    Fs = []
    for i in range(n_H):
        Fs.append(makeF(M, As[i], Bs[i], lX))

    a2t = np.full((len(lX) - 1), 0.0, dtype=float)
    for it in range(len(lX) - 1):
        a2t[it] = 0.5 * (lX[it] + lX[it + 1])
    a2x = np.full((len(lX) - 1), 0.0, dtype=float)
    for it in range(len(lX) - 1):
        a2x[it] = Fs[0](a2t[it])

    n_T = 300
    T_ln = np.linspace(min(Ts), max(Ts), n_T)
    X_ln = np.full((n_T), 0.0, dtype=float)
    for it in range(n_T):
        X_ln[it] = Fs[0](T_ln[it])
    dist_for_elevation = np.full((n_H, n_T), 0.0, dtype=float)
    for ih in range(n_H):
        for jt in range(n_T):
            dist_for_elevation[ih, jt] = Fs[ih](T_ln[jt])
    elev = np.full((n_T), 0.0, dtype=float)
    slopes = np.full((n_T), 0.0, dtype=float)
    for jt in range(n_T):
        p = np.polyfit(np.arange(0, n_H, 1), dist_for_elevation[:, jt], 1)
        slopes[jt] = p[0]
        elev[jt] = math.atan(p[0]) * 180.0 / math.pi

    # 描画
    fig = plt.figure()
    fig.suptitle("Cluster Analysis")
    # クラスタの点群の様子を3Dでプロット
    ax00 = fig.add_subplot(1, 3, 1, projection="3d")
    ax00.scatter(Ts, Xs, Hs, s=5, c="blue")
    ax00.set_xlabel("Time")
    ax00.set_ylabel("Distance")
    ax00.set_zlabel("Height")
    # 各高度・時刻での速度をプロット
    ax01 = fig.add_subplot(2, 3, 2)
    for ih in range(n_H):
        ax01.plot(a2t, As[ih], label="{i}".format(i=ih))
    ax01.set_xlabel("Time")
    ax01.set_ylabel("Velocity")
    ax01.legend()
    # 高度・位置ごとの速度をプロット
    ax02 = fig.add_subplot(2, 3, 3)
    for ih in range(n_H):
        ax02.plot(a2x, As[ih], label="{i}".format(i=ih))
    ax02.set_xlabel("Distance")
    ax02.set_ylabel("Velocity")
    ax02.legend()
    # 時刻ごとの仰角を調べる(一つの時刻tを固定して、その時刻での傾きを調べる)
    ax11 = fig.add_subplot(2, 3, 5)
    ax11.plot(T_ln, elev)
    ax11.set_xlabel("Time")
    ax11.set_ylabel("Zenith")
    # 位置ごとの仰角を調べる
    ax12 = fig.add_subplot(2, 3, 6)
    ax12.plot(X_ln, elev)
    ax12.set_xlabel("Distance")
    ax12.set_ylabel("Zenith")

    plt.show()


def Clustering(p):
    n_point = len(p)
    Z = linkage(p, method="single")
    t4 = 0.1
    Cl = fcluster(Z, t4, criterion="distance")
    n_group = max(Cl)
    t_list = [[] for i in range(n_group)]
    x_list = [[] for i in range(n_group)]
    h_list = [[] for i in range(n_group)]
    for i in range(n_point):
        t_list[Cl[i] - 1].append(p[i][0])
        x_list[Cl[i] - 1].append(p[i][1])
        h_list[Cl[i] - 1].append(p[i][2])
    return n_group, t_list, x_list, h_list


def SegmentRegression(Ts, Xs, dT, lam=1.0e3, Tbound=[np.nan, np.nan]):
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

    mT = 0.0
    MT = 0.0
    if Tbound[0] == np.nan:
        mT = min(Ts)
    else:
        mT = Tbound[0]
    if Tbound[1] == np.nan:
        MT = max(Ts)
    else:
        MT = Tbound[1]

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


# 3次元クラスタリング?
def Lag(
    drive: str,
    exps: list[EXPERIMENT],
    r_p: np.ndarray,  # End points of Line
    divs: np.ndarray,  # Height for calc
    threshold: float,
):
    country = exps[0].c
    year4 = exps[0].y
    day = exps[0].d
    code = exps[0].code

    rall, a2t, setting = ImportDTomoSequence(drive, exps)
    n_h = divs.shape[0]
    n_t = a2t.shape[0]
    s_b = r_p[0, 0]
    s_l = r_p[0, 1]
    e_b = r_p[1, 0]
    e_l = r_p[1, 1]

    a1h, a1b, a1l = setting.plains()

    m_H = 0.0

    n_p = 100
    bs = np.linspace(s_b, e_b, n_p)
    ls = np.linspace(s_l, e_l, n_p)

    dist = np.full((n_p), 0.0, dtype=float)
    for ip in range(n_p):
        dist[ip] = L2_on_ellipsoid(
            phi_1=bs[0], L_1=ls[0], h_1=m_H, phi_2=bs[ip], L_2=ls[ip], h_2=m_H
        )

    time = np.full((n_t), 0.0, dtype=float)
    for it in range(n_t):
        time[it] = a2t[it] / 120.0

    hgts = np.full((n_h), 0.0, dtype=float)
    for ih in range(n_h):
        hgts = a1h[divs[ih]]

    res = np.full((n_h, n_t, n_p), np.nan, dtype=float)

    for ih in range(n_h):
        for jt in range(n_t):
            for kp in range(n_p):
                lat = bs[kp]
                lon = ls[kp]
                B = lnearst(a1b, lat)
                L = lnearst(a1l, lon)
                res[ih, jt, kp] = rall[jt, ih, B, L]

    ind = np.full((n_h, n_t, n_p), 0, dtype=int)
    for ih in range(n_h):
        for jt in range(1, n_t - 1):
            for kp in range(n_p):
                if np.isnan(res[ih, jt, kp]):
                    pass
                else:
                    if res[ih, jt, kp] < -threshold:
                        ind[ih, jt, kp] = -1
                    elif res[ih, jt, kp] > threshold:
                        ind[ih, jt, kp] = 1

    d_var = np.mean(dist)
    d_sigma = np.std(dist)
    h_var = np.mean(hgts)
    h_sigma = np.std(hgts)

    dist = zscore(dist)
    hgts = zscore(hgts)

    x_minus = []
    h_minus = []
    t_minus = []
    p_minus = []
    x_plus = []
    h_plus = []
    t_plus = []
    p_plus = []

    for ih in range(n_h):
        for jt in range(n_t):
            for kp in range(n_p):
                if ind[ih, jt, kp] < 0:
                    X = dist[kp]
                    H = hgts[ih]
                    T = time[jt]
                    x_minus.append(X)
                    h_minus.append(H)
                    t_minus.append(T)
                    p_minus.append([T, X, H])
                if ind[ih, jt, kp] > 0:
                    X = dist[kp]
                    H = hgts[ih]
                    T = time[jt]
                    x_plus.append(X)
                    h_plus.append(H)
                    t_plus.append(T)
                    p_plus.append([T, X, H])

        n_minus, gt_minus, gx_minus, gh_minus = Clustering(p_minus)

        for i_minus in range(n_minus):
            for jx in range(len(gx_minus[i_minus])):
                gx_minus[i_minus][jx] = gx_minus[i_minus][jx] * d_sigma + d_var
            for jt in range(len(gt_minus[i_minus])):
                gt_minus[i_minus][jx] = gt_minus[i_minus][jx] * h_sigma + h_var

        for i_minus in range(n_minus):
            _3D_SegmentRegression(
                gt_minus[i_minus], gx_minus[i_minus], gh_minus[i_minus], 0.5
            )


if __name__ == "__main__":
    drive = "D:"
    country = "jp"
    year4 = 2016
    day = 193
    code = "MSTID_1+2_04d_3_2_0_8-1"
    exps = []
    for i in range(1320, 2200, 1):
        exps.append(EXPERIMENT(country, year4, day, code, i))
    Lag(
        drive,
        exps,
        np.array([[50.0, 150.0], [25.0, 125.0]]),
        np.arange(3, 10, 1),
        threshold=0.003,
    )
