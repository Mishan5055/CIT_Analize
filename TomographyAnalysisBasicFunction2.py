import numpy as np
from TomographyAnalysisBasicFunction import SETTING, EXPERIMENT
from TomographyAnalysisBasicFunction import (
    lnearst,
    ImportDTomoSequence,
    L2_on_ellipsoid,
    ExtractDataOnLine,
    ExtractLocalStationary,
)

from _SegmentRegression import _3D_SegmentRegression, findex

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


def Clustering(p):
    n_point = len(p)
    Z = linkage(p, method="single")
    t4 = 0.05
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


def is_TID(t, x, h):
    if len(t) < 10:
        return False
    mT = min(t)
    MT = max(t)
    print(mT, "-", MT)
    if MT - mT < 1.0:
        return False
    else:
        return True


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
        hgts[ih] = a1h[divs[ih]]
    print(hgts)

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

    dist = zscore(dist)

    for ih in range(n_h):
        hgts[ih] *= 1.0e-3

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

    print(len(p_minus))
    # n_minus ... グループ数
    # gt_minus ... 各クラスタの時刻tのリスト
    # gx_minus ... 各クラスタの距離xのリスト
    # gh_minus ... 各クラスタの高度hのリスト
    n_minus, gt_minus, gx_minus, gh_minus = Clustering(p_minus)

    for i_minus in range(n_minus):
        for jx in range(len(gx_minus[i_minus])):
            gx_minus[i_minus][jx] = gx_minus[i_minus][jx] * d_sigma + d_var
        for jh in range(len(gh_minus[i_minus])):
            gh_minus[i_minus][jh] = gh_minus[i_minus][jh] * 1.0e3

    for i_minus in range(n_minus):
        print(i_minus, "/", n_minus, ":", len(gt_minus[i_minus]))
        if not is_TID(gt_minus[i_minus], gx_minus[i_minus], gh_minus[i_minus]):
            continue
        st_minus = []
        sx_minus = []
        sh_minus = []
        for j_point in range(len(gt_minus[i_minus])):
            if findex(st_minus, gt_minus[i_minus][j_point]) == -1:
                st_minus.append(gt_minus[i_minus][j_point])
            if findex(sx_minus, gx_minus[i_minus][j_point]) == -1:
                sx_minus.append(gx_minus[i_minus][j_point])
            if findex(sh_minus, gh_minus[i_minus][j_point]) == -1:
                sh_minus.append(gh_minus[i_minus][j_point])
        _3D_SegmentRegression(
            exps,
            gt_minus[i_minus],
            gx_minus[i_minus],
            gh_minus[i_minus],
            st_minus,
            sx_minus,
            sh_minus,
            0.5,
            "minus_{i:03d}".format(i=i_minus),
        )

    n_plus, gt_plus, gx_plus, gh_plus = Clustering(p_plus)

    for i_plus in range(n_plus):
        for jx in range(len(gx_plus[i_plus])):
            gx_plus[i_plus][jx] = gx_plus[i_plus][jx] * d_sigma + d_var
        for jh in range(len(gh_plus[i_plus])):
            gh_plus[i_plus][jh] = gh_plus[i_plus][jh] * 1.0e3

    for i_plus in range(n_plus):
        print(i_plus, "/", n_plus, ":", len(gt_plus[i_plus]))
        if not is_TID(gt_plus[i_plus], gx_plus[i_plus], gh_plus[i_plus]):
            continue
        st_plus = []
        sx_plus = []
        sh_plus = []
        for j_point in range(len(gt_plus[i_plus])):
            if findex(st_plus, gt_plus[i_plus][j_point]) == -1:
                st_plus.append(gt_plus[i_plus][j_point])
            if findex(sx_plus, gx_plus[i_plus][j_point]) == -1:
                sx_plus.append(gx_plus[i_plus][j_point])
            if findex(sh_plus, gh_plus[i_plus][j_point]) == -1:
                sh_plus.append(gh_plus[i_plus][j_point])
        _3D_SegmentRegression(
            exps,
            gt_plus[i_plus],
            gx_plus[i_plus],
            gh_plus[i_plus],
            st_plus,
            sx_plus,
            sh_plus,
            0.5,
            "plus_{i:03d}".format(i=i_plus),
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
        np.arange(5, 14, 1),
        threshold=0.002,
    )
