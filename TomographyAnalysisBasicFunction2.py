
from TomographyAnalysisBasicFunction import EXPERIMENT, SETTING
from TomographyAnalysisBasicFunction import L2_on_ellipsoid, ImportDTomoSequence, lnearst

import numpy as np
from scipy.stats import zscore
from scipy.cluster.hierarchy import linkage, fcluster
import os
import matplotlib.pyplot as plt
from math import atan, pi

from mpl_toolkits.mplot3d import Axes3D


def LSP(x, y, z) -> np.ndarray:
    """
        Returns : \n
        z = ax+by+c \n
        a,b,c \n
    """
    M = len(x)
    A = np.full((M, 3), 0.0, dtype=float)
    Z = np.full((M), 0.0, dtype=float)
    for i in range(M):
        A[i, 0] = x[i]
        A[i, 1] = y[i]
        A[i, 2] = 1.0
        Z[i] = z[i]
    param = np.linalg.solve(A.T@A, A.T@Z)
    return param

# r_p ... End point coordinates of a line (2x2)
# r_p[0,:] ... (lat,lon) of start point
# r_p[1,:] ... (lat,lon) of end point


def Lag(drive: str, exps: list[EXPERIMENT], h_idx: np.ndarray, r_p: np.ndarray, threshold=0.003, lwindow=121):
    """
    _summary_
    3次元クラスタリングを行い、TIDについての解析をします。\n
    Given : drive : str,データのあるドライブを指定します \n
            exps : list[EXPERIMENT],読み込むデータのEXPERIMENTオブジェクトをリストで指定します \n
            hgts : \n
            r_p : 考える直線の端点を緯度経度で指定します。例えば、(25N,125E)と(50N,150E)を端点とする場合、\n
                    r_p = np.array( [[25.0,125.0],[50.0,150.0]]) \n
                  と指定してください。 \n
            threshold : float, 異常とする境界値を設定します \n
            lwindow : int,デトレンドする際のウィンドウ幅を指定します
    """

    country = exps[0].c
    year4 = exps[0].y
    day = exps[0].d
    code = exps[0].code

    datas, a2t, setting = ImportDTomoSequence(drive, exps, lwindow)

    n_t = a2t.shape[0]
    n_h = h_idx.shape[0]
    s_b = r_p[0, 0]
    s_l = r_p[0, 1]
    e_b = r_p[1, 0]
    e_l = r_p[1, 1]

    a1h, a1b, a1l = setting.plains()

    m_H = a1h[h_idx[0]]

    hgts = np.full((n_h), 0.0, dtype=float)

    for ih, hgt in enumerate(h_idx):
        hgts[ih] = setting.a2h[hgt]

    n_p = 100
    bs = np.linspace(s_b, e_b, n_p)
    ls = np.linspace(s_l, e_l, n_p)

    dist = np.full((n_p), 0.0, dtype=float)

    for ip in range(n_p):
        dist[ip] = L2_on_ellipsoid(
            phi_1=bs[0], L_1=ls[0], h_1=m_H, phi_2=bs[ip], L_2=ls[ip], h_2=m_H
        )

    d_var = np.mean(dist)
    d_std = np.std(dist)

    dist = zscore(dist)

    h_var = np.mean(hgts)
    h_std = np.std(hgts)

    hgts = zscore(hgts)

    time = np.full((n_t), 0.0, dtype=float)
    for it in range(n_t):
        time[it] = a2t[it]/120.0

    res = np.full((n_h, n_t, n_p), np.nan, dtype=float)

    for hp in range(n_p):
        lat = bs[hp]
        lon = ls[hp]
        B = lnearst(a1b, lat)
        L = lnearst(a1l, lon)
        for ih in range(n_h):
            for jt in range(n_t):
                res[ih, jt, hp] = datas[jt, h_idx[ih], B, L]

    t_minus = []
    x_minus = []
    h_minus = []
    p_minus = []

    for ih in range(n_h):
        H = hgts[ih]
        for jt in range(n_t):
            T = time[it]
            for kp in range(n_p):
                if res[ih, jt, kp] < -1.0*threshold:
                    X = dist[kp]
                    x_minus.append(X)
                    t_minus.append(T)
                    h_minus.append(H)
                    p_minus.append([T, X, H])

    Z_minus = linkage(p_minus, method="single")

    t4 = 0.1
    c_minus4 = fcluster(Z_minus, t4, criterion="distance")

    print(max(c_minus4), len(c_minus4))
    os.makedirs("C:/v_tid/{c}/{y4:04d}/{d:03d}/{cd}".format(
        c=country, y4=year4, d=day, cd=code
    ))

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(t_minus, x_minus, h_minus, s=3, c=c_minus4, cmap="jet")
    ax.set_xlabel("Time [UT/hour]")
    ax.set_ylabel("Distance [km]")
    ax.set_zlabel("Height [km]")

    fig.savefig("C/v_tid/{c}/{y4:04d}/{d:03d}/{cd}/minus_clusters.png".format(
        c=country, y=year4, d=day, cd=code
    ))

    plt.clf()
    plt.close()

    n_group = max(c_minus4)
    n_point = len(p_minus)

    for hg in range(n_group):
        group_t = []
        group_x = []
        group_h = []
        group_p = []
        is_TID = False
        for ip in range(n_point):
            if c_minus4[ip] == hg+1:
                T = t_minus[ip]
                X = x_minus[ip]*d_std+d_var
                H = h_minus[ip]*h_std+h_var
                group_t.append(T)
                group_x.append(X)
                group_h.append(H)
                group_p.append([T, X, H])

        if max(group_t)-min(group_t) > 1.0:
            is_TID = True
        if is_TID:
            param = LSP(group_t, group_x, group_h)
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            ax.scatter(group_t, group_x, group_h, s=3, c="black")
            ax.set_xlabel("T [UT/hour]")
            ax.set_ylabel("X [Distance/km]")
            ax.set_zlabel("Z [Height/km]")
            ax.set_title(
                "a={aa:06.4f} b={bb:06.4f} c={cc:06.4f} : v={v:06.4f} d(delay)={d:06.4f} phi={p:06.4f}".format(
                    aa=param[0], bb=param[1], cc=param[2], v=-
                    param[0]/param[1],
                    d=1.0/param[0], p=atan(param[1])*180.0/pi
                )
            )
            ax.set_xlim(0, 2500)
            ax.set_ylim(11, 19)
            ax.set_zlim(200, 400)
            fig.savefig("C:/v_tid/{c}/{y4:04d}/{d:03d}/{cd}/minus_{h:03d}".format(
                c=country, y=year4, d=day, cd=code,
            ), dpi=100)
            plt.clf()
            plt.close()


if __name__ == "__main__":
    drive = "E:"
    country = "jp"
    year4 = 2016
    day = 193
    code = "MSTID_1+2_05d_3_2_0_8-1"

    exps = []
    for ep in range(1600, 2000, 1):
        exp = EXPERIMENT(country, year4, day, code, ep)
        exps.append(exp)

    hgts = np.arange(2, 15, 1)
    r_p = np.array([[25.0, 125.0], [50.0, 150.0]])

    Lag(drive, exps, hgts, r_p)
