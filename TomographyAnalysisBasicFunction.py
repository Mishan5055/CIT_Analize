import os
from tqdm import tqdm
from mpl_toolkits.basemap import Basemap
import numpy as np
from matplotlib import pyplot as plt
import math
import datetime
from math import acos, cos, sin
from scipy.sparse import csr_matrix

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import zscore

import matplotlib

rf = 1.0 / 298.257223563
ra = 6378.1370
rb = ra * (1.0 - rf)
re = math.sqrt((ra * ra - rb * rb) / (ra * ra))
drive = "D:"
# matplotlib.use("Agg")


def lnearst(l: np.ndarray, v: float):
    L = len(l)
    if l[0] > v:
        return -1
    for i in range(L):
        if l[i] > v:
            return i - 1
        if i == L - 1:
            return L - 1


class BLH:
    # b,l...[degree]
    # h...[km]
    b: float = 0.0
    l: float = 0.0
    h: float = 0.0

    def __init__(self, b, l, h):
        self.b = b
        self.l = l
        self.h = h

    def to_XYZ(self):
        answer = XYZ(0.0, 0.0, 0.0)
        n = ra / math.sqrt(
            1.0
            - re * re * math.sin(math.radians(self.b)) * math.sin(math.radians(self.b))
        )
        answer.x = (
            (n + self.h)
            * math.cos(math.radians(self.b))
            * math.cos(math.radians(self.l))
        )
        answer.y = (
            (n + self.h)
            * math.cos(math.radians(self.b))
            * math.sin(math.radians(self.l))
        )
        answer.z = ((1 - re * re) * n + self.h) * math.sin(math.radians(self.b))
        return answer

    def __str__(self):
        return (
            "[ B: " + str(self.b) + " L: " + str(self.l) + " H: " + str(self.h) + " ]"
        )


class XYZ:
    # x,y,z...[km]
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def to_BLH(self):
        X = float(self.x)
        Y = float(self.y)
        Z = float(self.z)
        answer = BLH(0.0, 0.0, 0.0)
        # 1
        p = math.sqrt(X * X + Y * Y)
        h = ra * ra - rb * rb
        t = math.atan2(Z * ra, p * rb)  # rad
        answer.l = math.degrees(math.atan2(Y, X))  # deg
        # 2
        answer.b = math.degrees(
            math.atan2(
                (ra * rb * Z + ra * h * math.sin(t) ** 3),
                (ra * rb * p - rb * h * math.cos(t) ** 3),
            )
        )  # deg
        # 3
        n = ra / math.sqrt(
            1
            - re
            * re
            * math.sin(math.radians(answer.b))
            * math.sin(math.radians(answer.b))
        )
        # 4
        answer.h = p / math.cos(math.radians(answer.b)) - n
        return answer

    def __str__(self):
        return (
            "[ X: " + str(self.x) + " Y: " + str(self.y) + " Z: " + str(self.z) + " ]"
        )

    def __add__(self, other):
        spo = XYZ(self.x + other.x, self.y + other.y, self.z + other.z)
        return spo

    def __sub__(self, other):
        smo = XYZ(self.x - other.x, self.y - other.y, self.z - other.z)
        return smo

    def __mul__(self, other):
        sto = XYZ(self.x * other, self.y * other, self.z * other)
        return sto

    def __rmul__(self, other):
        sto = XYZ(self.x * other, self.y * other, self.z * other)
        return sto

    def L2(self) -> float:
        siz = self.x**2 + self.y**2 + self.z**2
        return math.sqrt(siz)


class EXPERIMENT:
    c: str = ""
    y: int = 0
    d: int = 0
    code: str = ""
    ep: int = 0

    def __init__(self, c, y, d, code, ep):
        self.c = c
        self.y = y
        self.d = d
        self.code = code
        self.ep = ep

    def __str__(self):
        return (
            self.c
            + " "
            + str(self.y)
            + " / "
            + str(self.d)
            + " "
            + self.code
            + " : "
            + str(self.ep)
        )


class SETTING:
    a1b: np.ndarray = np.array([])
    a1l: np.ndarray = np.array([])
    a1h: np.ndarray = np.array([])
    a2b: np.ndarray = np.array([])
    a2l: np.ndarray = np.array([])
    a2h: np.ndarray = np.array([])

    n_h: int = 0
    n_b: int = 0
    n_l: int = 0
    n_all: int = 0
    cof: float = 1.0e1
    alpha: float = 0.3

    m_H: float = 0.0
    M_H: float = 0.0
    m_B: float = 0.0
    M_B: float = 0.0
    m_L: float = 0.0
    M_L: float = 0.0

    H: csr_matrix = csr_matrix((1, 1))
    trH_2: float = 0.0

    def __init__(self, a1h, a1b, a1l, cof, alpha):
        self.a1h = a1h
        self.a1b = a1b
        self.a1l = a1l
        self.cof = cof
        self.alpha = alpha

        self.n_h = a1h.shape[0] - 1
        self.n_b = a1b.shape[0] - 1
        self.n_l = a1l.shape[0] - 1
        self.n_all = self.n_h * self.n_b * self.n_l

        self.a2h = np.full((self.n_h), 0.0, dtype=float)
        self.a2b = np.full((self.n_b), 0.0, dtype=float)
        self.a2l = np.full((self.n_l), 0.0, dtype=float)
        for ih in range(self.n_h):
            self.a2h[ih] = 0.5 * (self.a1h[ih] + self.a1h[ih + 1])
        for ib in range(self.n_b):
            self.a2b[ib] = 0.5 * (self.a1b[ib] + self.a1b[ib + 1])
        for il in range(self.n_l):
            self.a2l[il] = 0.5 * (self.a1l[il] + self.a1l[il + 1])

        self.m_H = np.min(self.a1h)
        self.M_H = np.max(self.a1h)
        self.m_B = np.min(self.a1b)
        self.M_B = np.max(self.a1b)
        self.m_L = np.min(self.a1l)
        self.M_L = np.max(self.a1l)

    def __initH__(self, H: csr_matrix):
        self.H = H
        self.trH_2 = csr_matrix.trace(self.H.T * self.H)

    def nbrock(self) -> tuple[int, int, int, int]:
        """_summary_
        settingのブロック数を返します。 \n
        return : n_all, n_h, n_b, n_l \n
        Returns: \n
            tuple[int, int, int, int]: _description_
        """
        return [self.n_all, self.n_h, self.n_b, self.n_l]

    def plains(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """_summary_
        settingに設定された境界を返します。\n
        return : a1h, a1b, a1l \n
        Returns: \n
            tuple[np.ndarray, np.ndarray, np.ndarray]: _description_
        """
        return [self.a1h, self.a1b, self.a1l]

    def centers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return [self.a2h, self.a2b, self.a2l]

    def bounds(self) -> tuple[float, float, float, float, float, float]:
        return [self.m_H, self.M_H, self.m_B, self.M_B, self.m_L, self.M_L]

    def simprize(self, N_b, N_l):
        l2b = np.full((self.n_b), 0.0, dtype=float)
        l2l = np.full((self.n_l), 0.0, dtype=float)
        for ib in range(self.n_b):
            l2b[ib] = self.a1b[ib + 1] - self.a1b[ib]
        for il in range(self.n_l):
            l2l[il] = self.a1l[il + 1] - self.a1l[il]
        b2b = np.full((self.n_b + 1), True, dtype=bool)
        b2l = np.full((self.n_l + 1), True, dtype=bool)
        loop = 0
        while np.sum(b2b) > N_b:
            argMIN = -1
            MIN = 100.0
            lower = 0
            upper = 1
            while upper < self.n_b + 1:
                if b2b[upper]:
                    dif = self.a1b[upper] - self.a1b[lower]
                    if dif < MIN:
                        MIN = dif
                        argMIN = upper
                    lower = upper
                upper += 1
            b2b[argMIN] = False
            # print(loop, end=": ")
            # for ib in range(self.n_b+1):
            #     if b2b[ib]:
            #         print(self.a1b[ib], end=" ")
            # print("")
            loop += 1
        loop = 0
        while np.sum(b2l) > N_l:
            argMIN = -1
            MIN = 100.0
            lower = 0
            upper = 1
            while upper < self.n_l + 1:
                if b2l[upper]:
                    dif = self.a1l[upper] - self.a1l[lower]
                    if dif < MIN:
                        MIN = dif
                        argMIN = upper
                    lower = upper
                upper += 1
            b2l[argMIN] = False
            # print(loop, end=": ")
            # for il in range(self.n_l+1):
            #     if b2l[il]:
            #         print(self.a1l[il], end=" ")
            # print("")
            loop += 1
        simple_a1b = []
        simple_a1l = []
        for ib in range(self.n_b + 1):
            if b2b[ib]:
                simple_a1b.append(self.a1b[ib])
        for il in range(self.n_l + 1):
            if b2l[il]:
                simple_a1l.append(self.a1l[il])

        simprize_prob = SETTING(
            self.a1h, np.array(simple_a1b), np.array(simple_a1l), self.cof, 0.3
        )
        return simprize_prob

    def extension(self, simple, X) -> np.ndarray:
        X_0 = np.full((self.n_all), 0.0, dtype=float)
        sa1h, sa1b, sa1l = simple.plains()
        for ih in range(self.n_h):
            sh = lnearst(sa1h, self.a2h[ih])
            for jb in range(self.n_b):
                sb = lnearst(sa1b, self.a2b[jb])
                for kl in range(self.n_l):
                    sl = lnearst(sa1l, self.a2l[kl])
                    idx = kl + jb * self.n_l + ih * self.n_b * self.n_l
                    sidx = sl + sb * simple.n_l + sh * simple.n_b * simple.n_l
                    X_0[idx] = X[sidx]

        return X_0


class INPUT:
    tec: np.ndarray = np.array([])
    sat: list[XYZ] = []
    rec: list[XYZ] = []

    n_obs: int = 0

    def __init__(self, tec, sat, rec):
        self.tec = tec
        self.sat = sat
        self.rec = rec
        self.n_obs = self.tec.shape[0]


# import .tomo file


def ImportTomo(
    drive, exp: EXPERIMENT
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SETTING]:
    country = exp.c
    year4 = exp.y
    day = exp.d
    code = exp.code
    epoc = exp.ep

    n_h = 0
    n_b = 0
    n_l = 0

    Tomography_Output = "{dr}/tomo/{c}/{y:04d}/{d:03d}/{cd}/{ep:04d}.tomo".format(
        dr=drive, c=country, y=year4, d=day, cd=code, ep=epoc
    )
    A_Output = "{dr}/tomoc/{c}/{y:04d}/{d:03d}/{cd}/{ep:04d}.tomoc".format(
        dr=drive, c=country, y=year4, d=day, cd=code, ep=epoc
    )

    with open(Tomography_Output, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if "END OF HEADER" in line:
                break
            if "Number of Boxel (All area)" in line:
                line = f.readline()
                n_b = int(line.split()[3])
                line = f.readline()
                n_l = int(line.split()[3])
                line = f.readline()
                n_h = int(line.split()[3])
                a1b = np.full((n_b + 1), 0.0, dtype=float)
                a1l = np.full((n_l + 1), 0.0, dtype=float)
                a1h = np.full((n_h + 1), 0.0, dtype=float)
            if "Plain List" in line:
                line = f.readline()  # ... # Latitude
                for ib in range(n_b + 1):
                    line = f.readline()
                    a1b[ib] = float(line.split()[1])
                line = f.readline()  # ... # Longitude
                for il in range(n_l + 1):
                    line = f.readline()
                    a1l[il] = float(line.split()[1])
                line = f.readline()  # ... # Altitude
                for ih in range(n_h + 1):
                    line = f.readline()
                    a1h[ih] = float(line.split()[1])
                setting = SETTING(a1h, a1b, a1l, 0.0, 0.0)

        nall, nh, nb, nl = setting.nbrock()

        datas = np.full((nh, nb, nl), 0.0, dtype=float)
        for kh in range(nh):
            line = f.readline()
            for jb in range(nb):
                line = f.readline()
                for il in range(nl):
                    datas[kh, jb, il] = float(line.split()[il])

    A_bool = np.full((nh, nb, nl), 0, dtype=int)
    with open(A_Output, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            else:
                idx = int(line.split()[1])
                l = idx % nl
                idx = (idx - l) // nl
                b = idx % nb
                idx = (idx - b) // nb
                h = idx % nh
                A_bool[h, b, l] += 1

    result = np.full((nh, nb, nl), np.nan, dtype=float)
    for ih in range(nh):
        for jb in range(nb):
            for kl in range(nl):
                if A_bool[ih, jb, kl] > 0:
                    result[ih, jb, kl] = datas[ih, jb, kl]

    return datas, A_bool, result, setting


# import .tomo file(s). specify files by list epochs
# returns
# all_datas : np.ndarray(float) : raw data
# all_C     : np.ndarray(int)   : number of path data
# all_result: np.ndarray(float) : raw data considering missing value
# a1h,a1b,a1l


def ImportTomoSequence(drive, exps: list[EXPERIMENT]):
    """

    一連の.tomoファイルを読み込みます。\n

    Args:\n
        drive (_type_): ファイルのあるドライブ \n
        exps (list[EXPERIMENT]): ファイル群を指定するEXPERIMENTのリスト \n

    Returns: \n
        datas: 欠損値を考慮しない結果 \n
        C: 各ブロックに入り込むパスの数 \n
        result: 欠損値を考慮した結果 \n
        setting:
    """
    n_t = len(exps)
    test, a_test, res_test, setting = ImportTomo(drive, exps[0])
    n_all, n_h, n_b, n_l = setting.nbrock()

    all_datas = np.full((n_t, n_h, n_b, n_l), 0.0, dtype=float)
    all_C = np.full((n_t, n_h, n_b, n_l), 0, dtype=int)
    all_result = np.full((n_t, n_h, n_b, n_l), np.nan, dtype=float)

    print("loading result files ...")
    for iepoch, epoch in tqdm(enumerate(exps)):
        datas, A_bool, result, setting = ImportTomo(drive, exps[iepoch])
        all_datas[iepoch] = datas
        all_C[iepoch] = A_bool
        all_result[iepoch] = result

    print(">>> loading result summary <<<")
    print("Number of files : {nt:03d}".format(nt=n_t))
    print(
        "Filling rate    : {fr:5.3f}".format(
            fr=np.sum(all_C > 0) / (n_t * n_h * n_b * n_l)
        )
    )

    return all_datas, all_C, all_result, setting


# Detrended Ne


def detrended_TEC(drive, exps, setting: SETTING, rall, lwindow=121, record=True):
    """

    Args: \n
        drive (_type_): データを書き出すドライブを指定 \n
        exps (_type_): EXPのリストを指定 \n
        setting (SETTING):  \n
        rall (_type_): 欠損値を考慮したデータ \n
        lwindow (int, optional): ウィンドウ幅 Defaults to 121. \n
        record (bool, optional): 結果をファイルに記録するか Defaults to True.  \n

    Returns: \n
        detrended: デトレンドした結果 \n
    """
    country = exps[0].c
    year4 = exps[0].y
    day = exps[0].d
    code = exps[0].code

    n_t, n_h, n_b, n_l = rall.shape
    a1h, a1b, a1l = setting.plains()
    hwindow = math.floor(lwindow / 2)

    detrended_data = np.full((n_t, n_h, n_b, n_l), np.nan, dtype=float)

    for hepoch in range(hwindow, n_t - hwindow):
        subrdata = rall[hepoch - hwindow : hepoch + hwindow, :, :, :]
        for ih in range(n_h):
            for jb in range(n_b):
                for kl in range(n_l):
                    if not np.isnan(rall[hepoch, ih, jb, kl]):
                        # print(hepoch, ih, jb, kl)
                        detrended_data[hepoch, ih, jb, kl] = rall[
                            hepoch, ih, jb, kl
                        ] - np.nanmean(subrdata[:, ih, jb, kl])

    print(">>> summary <<<")
    print(
        "Filling rate    : {fr:5.3f}".format(
            fr=1.0 - np.sum(np.isnan(detrended_data)) / (n_t * n_h * n_b * n_l)
        )
    )
    print("Max value :", np.nanmax(np.abs(detrended_data)))

    if record:
        os.makedirs(
            "{dr}/dtomo/{c}/{y:04d}/{d:03d}/{cd}/".format(
                dr=drive, c=country, y=year4, d=day, cd=code
            ),
            exist_ok=True,
        )
        for hep, exp in enumerate(exps):
            epoch = exp.ep
            fdtomo = (
                "{dr}/dtomo/{c}/{y:04d}/{d:03d}/{cd}/{ep:04d}.dtomoh{lw:03d}".format(
                    dr=drive, c=country, y=year4, d=day, cd=code, ep=epoch, lw=lwindow
                )
            )
            with open(fdtomo, "w") as f:
                print("# Detrended Tomogeraphy Result", file=f)
                print("# ", file=f)
                print("# UTC : {dt}".format(dt=datetime.datetime.now()), file=f)
                print("# ", file=f)
                print("# Number of Boxel (All area)", file=f)
                print("# Latitude : {nb:03d}".format(nb=n_b), file=f)
                print("# Longitude : {nl:03d}".format(nl=n_b), file=f)
                print("# Height : {nh:03d}".format(nh=n_h), file=f)
                print("# ", file=f)
                print("# *** Plain List ***", file=f)
                print("# Latitude", file=f)
                for ib in range(n_b + 1):
                    print("# {b:+06.2f}".format(b=a1b[ib]), file=f)
                print("# Longitude", file=f)
                for il in range(n_l + 1):
                    print("# {l:+07.2f}".format(l=a1l[il]), file=f)
                print("# Height", file=f)
                for ih in range(n_h + 1):
                    print("# {h:+07.2f}".format(h=a1h[ih]), file=f)
                print("# ", file=f)
                print("# END OF HEADER", file=f)
                print("", file=f)

                for ih in range(n_h):
                    for jb in range(n_b):
                        for kl in range(n_l):
                            if np.isnan(detrended_data[hep, ih, jb, kl]):
                                print(
                                    "{dn:+13.11f} ".format(
                                        dn=detrended_data[hep, ih, jb, kl]
                                    ),
                                    end="",
                                    file=f,
                                )
                            else:
                                print(
                                    "{dn:+12.10f} ".format(
                                        dn=detrended_data[hep, ih, jb, kl]
                                    ),
                                    end="",
                                    file=f,
                                )
                        print("", file=f)
                    print("", file=f)

    return detrended_data


def ImportDTomo(drive, exp: EXPERIMENT, lwindow=121) -> tuple[np.ndarray, SETTING]:
    country = exp.c
    year4 = exp.y
    day = exp.d
    code = exp.code
    epoch = exp.ep

    n_b = 0
    n_h = 0
    n_l = 0
    fdtomo = "{dr}/dtomo/{c}/{y:04d}/{d:03d}/{cd}/{ep:04d}.dtomoh{lw:03d}".format(
        dr=drive, c=country, y=year4, d=day, cd=code, ep=epoch, lw=lwindow
    )
    with open(fdtomo, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if "END OF HEADER" in line:
                break
            if "Number of Boxel (All area)" in line:
                line = f.readline()
                n_b = int(line.split()[3])
                line = f.readline()
                n_l = int(line.split()[3])
                line = f.readline()
                n_h = int(line.split()[3])
                a1b = np.full((n_b + 1), 0.0, dtype=float)
                a1l = np.full((n_l + 1), 0.0, dtype=float)
                a1h = np.full((n_h + 1), 0.0, dtype=float)
            if "Plain List" in line:
                line = f.readline()  # ... # Latitude
                for ib in range(n_b + 1):
                    line = f.readline()
                    a1b[ib] = float(line.split()[1])
                line = f.readline()  # ... # Longitude
                for il in range(n_l + 1):
                    line = f.readline()
                    a1l[il] = float(line.split()[1])
                line = f.readline()  # ... # Altitude
                for ih in range(n_h + 1):
                    line = f.readline()
                    a1h[ih] = float(line.split()[1])
                setting = SETTING(a1h, a1b, a1l, 0.0, 0.0)

        n_all, n_h, n_b, n_l = setting.nbrock()

        datas = np.full((n_h, n_b, n_l), np.nan, dtype=float)
        for kh in range(n_h):
            line = f.readline()
            for jb in range(n_b):
                line = f.readline()
                for il in range(n_l):
                    if line.split()[il] == "+nan":
                        continue
                    else:
                        datas[kh, jb, il] = float(line.split()[il])

        return datas, setting


def ImportDTomoSequence(drive, exps: list[EXPERIMENT], lwindow=121):
    """_summary_
    一連のdtomoファイルを読み込みます。\n
    Returns : datas, a2t, setting : _description
    """
    n_t = len(exps)
    a2t = np.full(n_t, 0, dtype=int)
    for it in range(n_t):
        a2t[it] = exps[it].ep

    country = exps[0].c
    year4 = exps[0].y
    day = exps[0].d
    code = exps[0].code
    ep0 = exps[0].ep

    datas, setting = ImportDTomo(drive, exps[0], lwindow)

    nall, n_h, n_b, n_l = setting.nbrock()

    all_datas = np.full((n_t, n_h, n_b, n_l), np.nan, dtype=float)
    all_datas[0] = datas

    print("Importing dtomo files ....")
    for it in tqdm(range(n_t)):
        datas, setting = ImportDTomo(drive, exps[it], lwindow)
        all_datas[it] = datas

    return all_datas, a2t, setting


# Calc L2 distance on earth ellipsoid
# (phi_1,L_1,h_1) ... lat,lon,hgt of start point
# (phi_2,L_2,h_2) ... lat,lon,hgt of end point
# Ref : https://orsj.org/wp-content/corsj/or60-12/or60_12_701.pdf
# Accuracy <= 0.5%


def L2_on_ellipsoid(phi_1, L_1, h_1, phi_2, L_2, h_2):
    RA = 6370.0  # [km]
    phi_1 = phi_1 * math.pi / 180.0
    phi_2 = phi_2 * math.pi / 180.0
    L_1 = L_1 * math.pi / 180.0
    L_2 = L_2 * math.pi / 180.0
    h = 0.5 * (h_1 + h_2)
    tmp = sin(phi_1) * sin(phi_2) + cos(phi_1) * cos(phi_2) * cos(L_1 - L_2)
    if tmp > 1.0:
        tmp = 1.0
    elif tmp < -1.0:
        tmp = -1.0
    # print(tmp)
    L = (RA + h) * acos(tmp)
    return L


# r_p ... End point coordinates of a line (2x2)
# r_p[0,:] ... (lat,lon) of start point
# r_p[1,:] ... (lat,lon) of end point


def ExtractDataOnLine(rall, a2t, setting: SETTING, hgt: int, r_p: np.ndarray, div=5):
    n_t = a2t.shape[0]
    s_b = r_p[0, 0]
    s_l = r_p[0, 1]
    e_b = r_p[1, 0]
    e_l = r_p[1, 1]

    a1h, a1b, a1l = setting.plains()

    H = 0.5 * (a1h[hgt] + a1h[hgt + 1])

    n_p = 1000
    bs = np.linspace(s_b, e_b, n_p)
    ls = np.linspace(s_l, e_l, n_p)
    dist = np.full((n_p), 0.0, dtype=float)
    for ip in range(n_p):
        dist[ip] = L2_on_ellipsoid(
            phi_1=bs[0], L_1=ls[0], h_1=H, phi_2=bs[ip], L_2=ls[ip], h_2=H
        )

    time = np.full((n_t), 0.0, dtype=float)
    for it in range(n_t):
        time[it] = a2t[it] / 120.0

    res = np.full((n_t, n_p), np.nan, dtype=float)

    fig, ax = plt.subplots(1, 2, squeeze=False)

    for jp in range(n_p):
        lat = bs[jp]
        lon = ls[jp]
        B = lnearst(a1b, lat)
        L = lnearst(a1l, lon)
        for it in range(n_t):
            res[it, jp] = rall[it, hgt, B, L]

    ds, ts = np.meshgrid(dist, time)

    mB = 25.0
    MB = 50.0
    mL = 125.0
    ML = 150.0

    m = Basemap(
        llcrnrlat=mB,
        llcrnrlon=mL,
        urcrnrlat=MB,
        urcrnrlon=ML,
        resolution="l",
        ax=ax[0][0],
    )
    m.drawcoastlines()
    m.drawmeridians(np.arange(mL, ML, 5), labels=[0, 0, 0, 1], fontsize=10)
    m.drawparallels(np.arange(mB, MB, 5), labels=[1, 0, 0, 0], fontsize=10)

    xx, yy = m(r_p[:, 1], r_p[:, 0])

    ax[0][0].plot([xx[0], xx[1]], [yy[0], yy[1]], "k-", linewidth=1)
    ax[0][0].plot(xx, yy, "ko", markersize=5)

    ax[0][1].pcolormesh(
        ds, ts, res, cmap="jet", vmin=-0.01, vmax=0.01, shading="nearest"
    )
    ax[0][1].set_xlabel("distance [km]")
    ax[0][1].set_ylabel("time [UTC/hour]")

    plt.show()

    return res, ds, ts


def ExtractLocalStationary(res, ds, ts):
    n_t = res.shape[0]
    n_p = res.shape[1]
    ind = np.full((n_t, n_p), 0, dtype=int)

    for ip in range(n_p):
        for jt in range(1, n_t - 1):
            if np.isnan(res[jt, ip]):
                continue
            else:  # all not nan
                if res[jt, ip] < -0.003:
                    ind[jt, ip] = -1
                if res[jt, ip] > 0.003:
                    ind[jt, ip] = 1

    fig, ax = plt.subplots(1, 2, squeeze=False)
    ax[0][0].pcolormesh(
        ds, ts, res, cmap="jet", vmin=-0.01, vmax=0.01, shading="nearest"
    )
    ax[0][0].set_xlabel("distance [km]")
    ax[0][0].set_ylabel("time [UTC/hour]")
    ax[0][1].pcolormesh(ds, ts, ind, cmap="bwr", vmin=-1, vmax=1, shading="nearest")
    ax[0][1].set_xlabel("distance [km]")
    ax[0][1].set_ylabel("time [UTC/hour]")

    plt.show()

    return ind


def Cluster(
    ind: np.ndarray, ds: np.ndarray, ts: np.ndarray, drive, country, year4, day, code
):
    # クラスタリングのためにそれぞれの軸で標準化
    dist = ds[0, :]
    time = ts[:, 0]

    d_var = np.mean(dist)
    d_sigma = np.std(dist)
    t_var = np.mean(time)
    t_sigma = np.std(time)

    # print(dist, time)

    dist = zscore(dist)
    time = zscore(time)

    n_t = time.shape[0]
    n_p = dist.shape[0]
    # print(n_t, n_p)

    x_minus = []
    y_minus = []
    p_minus = []

    x_plus = []
    y_plus = []
    p_plus = []

    for it in tqdm(range(n_t)):
        for jp in range(n_p):
            if ind[it, jp] < 0:
                X = dist[jp]
                Y = time[it]
                x_minus.append(X)
                y_minus.append(Y)
                p_minus.append([X, Y])
            if ind[it, jp] > 0:
                X = dist[jp]
                Y = time[it]
                x_plus.append(X)
                y_plus.append(Y)
                p_plus.append([X, Y])

    Z_minus = linkage(p_minus, method="single")

    t4 = 0.05 * max(Z_minus[:, 2])
    c_minus4 = fcluster(Z_minus, t4, criterion="distance")

    print(max(c_minus4), len(c_minus4), len(p_minus))

    os.makedirs(
        "{dr}/tid/{c}/{y4:04d}/{d:03d}/{cd}".format(
            dr=drive, c=country, y4=year4, d=day, cd=code
        ),
        exist_ok=True,
    )

    fig, ax = plt.subplots(1, 1, figsize=(13, 10), squeeze=False)
    ax[0, 0].scatter(x_minus, y_minus, s=3, c=c_minus4, cmap="jet")

    fig.savefig(
        "{dr}/tid/{c}/{y4:04d}/{d:03d}/{cd}/clusters.png".format(
            dr=drive, c=country, y4=year4, d=day, cd=code
        ),
        dpi=100,
    )

    plt.clf()
    plt.close()

    n_group = max(c_minus4)
    n_point = len(p_minus)

    for hg in range(n_group):
        group_x = []
        group_y = []
        group_p = []
        is_TID = False
        for ip in range(n_point):
            if c_minus4[ip] == hg + 1:
                X = x_minus[ip] * d_sigma + d_var
                Y = y_minus[ip] * t_sigma + t_var
                group_x.append(X)
                group_y.append(Y)
                group_p.append([X, Y])
        if max(group_y) - min(group_y) > 1.0:
            is_TID = True
        fig, ax = plt.subplots(1, 1, figsize=(13, 10), squeeze=False)
        if is_TID:
            ax[0, 0].scatter(group_x, group_y, s=2, c="red")
        else:
            ax[0, 0].scatter(group_x, group_y, s=2, c="black")
        ax[0, 0].set_title("Group={g},Size={s}".format(g=hg + 1, s=len(group_p)))
        ax[0, 0].set_xlim(0, 2500)
        ax[0, 0].set_ylim(11, 19)
        fig.savefig(
            "{dr}/tid/{c}/{y4:04d}/{d:03d}/{cd}/c_{h:03d}.png".format(
                dr=drive, c=country, y4=year4, d=day, cd=code, h=hg
            ),
            dpi=100,
        )
        plt.clf()
        plt.close()


def draw_UD_records(
    drive, country, year4, day, code, epochs, a1h, a1b, a1l, rall, dtype="__"
):
    n_t = len(epochs)
    n_h = len(a1h) - 1
    MB = 50.0
    mB = 25.0
    ML = 150.0
    mL = 125.0
    MH = np.max(a1h)
    mH = np.min(a1h)

    lons, lats = np.meshgrid(a1l, a1b)
    print("Drawing records ...")
    for hepoch, epoch in tqdm(enumerate(epochs)):
        os.makedirs(
            "{dr}/v_tomo/{c}/{y4:04d}/{d:03d}/{cd}/{ep:04d}/{dt}_UD".format(
                dr=drive, c=country, y4=year4, d=day, cd=code, ep=epoch, dt=dtype
            ),
            exist_ok=True,
        )
        for ih in range(n_h):
            if np.all(np.isnan(rall[hepoch, ih, :, :])):
                continue
            fig, ax = plt.subplots(1, 1, figsize=(15, 15), squeeze=False)

            m = Basemap(
                llcrnrlon=mL, llcrnrlat=mB, urcrnrlon=ML, urcrnrlat=MB, resolution="l"
            )
            m.drawmeridians(
                np.arange(mL, ML, 5),
                color="gray",
                fontsize="small",
                labels=[False, False, False, True],
            )
            m.drawparallels(
                np.arange(mB, MB, 5),
                color="gray",
                fontsize="small",
                labels=[True, False, False, False],
            )
            m.drawcoastlines()
            im = m.pcolormesh(
                lons,
                lats,
                rall[hepoch, ih, :, :],
                cmap="jet",
                vmin=-0.01,
                vmax=0.01,
                shading="flat",
            )
            plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
            plt.colorbar(im, label="Density [TECU/km=1.0e+13[/m^3]]")
            plt.title(
                "Country:{c} Year:{y4:04d} DOY:{d:03d} CODE:{cd} Epoc:{ep:04d} Height:{h1:04d}-{h2:04d}[km]".format(
                    c=country,
                    y4=year4,
                    d=day,
                    cd=code,
                    ep=epoch,
                    h1=round(a1h[ih]),
                    h2=round(a1h[ih + 1]),
                )
            )
            fig.savefig(
                "{dr}/v_tomo/{c}/{y4:04d}/{d:03d}/{cd}/{ep:04d}/{dt}_UD/{ih:02d}.png".format(
                    dr=drive,
                    c=country,
                    y4=year4,
                    d=day,
                    cd=code,
                    ep=epoch,
                    dt=dtype,
                    ih=ih,
                )
            )

            plt.close()
            plt.clf()


if __name__ == "__main__":
    drive = "D:"
    country = "jp"
    year4 = 2016
    doy = 193
    code = "keq_1+1_05d_3_1_2_3-1"
    epochs = np.arange(1320, 2220, 1)

    exps = []
    for ep in epochs:
        exp = EXPERIMENT(country, year4, doy, code, ep)
        exps.append(exp)

    # datas, C, result, setting = ImportTomoSequence(drive, exps)

    # dne = detrended_TEC(drive, exps, setting, result)

    datas, a2t, setting = ImportDTomoSequence(drive, exps)

    res, ds, ts = ExtractDataOnLine(
        datas, a2t, setting, 7, np.array([[45, 145], [30, 130]])
    )

    ind = ExtractLocalStationary(res, ds, ts)

    code = "keq_1+1_05d_3_1_2_3-1_3-3_1-1"

    Cluster(ind, ds, ts, drive, country, year4, doy, code)
