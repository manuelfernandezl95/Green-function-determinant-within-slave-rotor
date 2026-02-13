"""
Non-local electron Green's function determinant in a nodal-line Mott insulator
within the slave-rotor framework.

Computes the 2×2 Green's function matrix G(k,ω) and its determinant for a single
high‑symmetry k‑point, using converged slave‑rotor parameters loaded from .mat files.

Transcribed from MATLAB and optimised with NumPy/Numba.
Author: Manuel Fernandez Lopez
Date: 2026
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from numba import njit
from tqdm import tqdm

# ----------------------------------------------------------------------
# Physical parameters & lattice definition
# ----------------------------------------------------------------------
t = 1.0
N1 = N2 = N3 = 80          # k‑mesh density
Ns = 2
Uv = np.linspace(0.0001, 5, 50)

# FCO lattice vectors (half unit cell)
a, b, c = 13.2150, 19.4783, 19.6506
d1 = np.array([0, b, c]) / 2
d2 = np.array([a, 0, c]) / 2
d3 = np.array([a, b, 0]) / 2

# Reciprocal lattice vectors
b1 = np.array([-2*np.pi/a,  2*np.pi/b,  2*np.pi/c])
b2 = np.array([ 2*np.pi/a, -2*np.pi/b,  2*np.pi/c])
b3 = np.array([ 2*np.pi/a,  2*np.pi/b, -2*np.pi/c])

# ----------------------------------------------------------------------
# k‑space grid (fully vectorised)
# ----------------------------------------------------------------------
def generate_k_grid(N1, N2, N3, b1, b2, b3):
    i = np.arange(N1)[:, None, None]
    j = np.arange(N2)[None, :, None]
    k = np.arange(N3)[None, None, :]
    kx = (i / N1) * b1[0] + (j / N2) * b2[0] + (k / N3) * b3[0]
    ky = (i / N1) * b1[1] + (j / N2) * b2[1] + (k / N3) * b3[1]
    kz = (i / N1) * b1[2] + (j / N2) * b2[2] + (k / N3) * b3[2]
    return kx, ky, kz

kx, ky, kz = generate_k_grid(N1, N2, N3, b1, b2, b3)

# ----------------------------------------------------------------------
# High‑symmetry path indices (exact replica of MATLAB logic)
# ----------------------------------------------------------------------
def generate_path_indices(N):
    """Return 0‑based index arrays P1, Q1, R1 for the k‑path."""
    N_mat = N
    ctr = N_mat // 2 + 1          # 41 for N=80 (1‑based)

    L1x = np.arange(ctr, 0, -5)           # 41,36,…,1
    M1 = len(L1x)
    L2x = np.arange(1, ctr+1, 5)          # 1,6,…,41
    L2z = np.ones_like(L2x)
    M2 = len(L2x)
    L3x = np.arange(ctr, N_mat+1, 5)      # 41,46,…,76
    L3y = np.ones_like(L3x) * ctr
    L3z = np.arange(1, ctr+1, 5)
    M3 = len(L3x)
    L4x = np.arange(ctr, 0, -10)          # 41,31,21,11,1
    L4y = np.arange(N_mat//4+1, 0, -5)    # 21,16,11,6,1
    M4 = len(L4x)
    L5x = np.arange(1, ctr+1, 5)
    L5y = np.ones_like(L5x)
    M5 = len(L5x)
    L6y = np.arange(1, ctr+1, 5)
    L6x = np.arange(ctr, N_mat+1, 5)
    L6z = np.ones_like(L5x) * ctr
    M6 = len(L6x)

    P1, Q1, R1 = [], [], []

    # Section 1
    for j in range(M1 - 1):
        P1.append(L1x[j])
        Q1.append(L1x[j])
        R1.append(L1x[j])
    # Section 2
    for j in range(M2 - 1):
        P1.append(L2x[j])
        Q1.append(L2x[j])
        R1.append(L2z[j])
    # Section 3
    for j in range(M3 - 1):
        P1.append(L3x[j])
        Q1.append(L3y[j])
        R1.append(L3z[j])
    # Section 4
    for j in range(M4 - 1):
        P1.append(L4x[j])
        Q1.append(L4y[j])
        R1.append(L4y[j])
    # Section 5
    for j in range(M5 - 1):
        P1.append(L5x[j])
        Q1.append(L5y[j])
        R1.append(L5x[j])
    # Section 6
    for j in range(M6):
        P1.append(L6x[j])
        Q1.append(L6y[j])
        R1.append(L6z[j])

    # Convert to 0‑based indexing
    return (np.array(P1, dtype=np.int32) - 1,
            np.array(Q1, dtype=np.int32) - 1,
            np.array(R1, dtype=np.int32) - 1)

P1, Q1, R1 = generate_path_indices(N1)

# ----------------------------------------------------------------------
# Load precomputed slave‑rotor parameters (adjust filenames if needed)
# ----------------------------------------------------------------------
data_Qfln = sio.loadmat('Qfaa.mat')['Qfln'].flatten()
data_Qftn = sio.loadmat('Qf1aa.mat')['Qftn'].flatten()
data_Qxln = sio.loadmat('Qxaa.mat')['Qxln'].flatten()
data_Qxtn = sio.loadmat('Qx1aa.mat')['Qxtn'].flatten()
data_lambdan = sio.loadmat('lambdaaa.mat')['lambdan'].flatten()
data_gapn   = sio.loadmat('RotorGapSOCaa.mat')['gapn'].flatten()
data_Zn     = sio.loadmat('quasiparticleaa.mat')['Zn'].flatten()

# Use the first U index from IUU = [29,33] (as in original code)
IUU = [29, 33]
ui = IUU[0] - 1
U = Uv[ui] / 2          # rescale as in MATLAB

Qfl = data_Qfln[ui]     # not used, kept for reference
Qft = data_Qftn[ui]
Qxl = data_Qxln[ui]     # not used
Qxt = data_Qxtn[ui]
Z   = data_Zn[ui]
lmbda = data_lambdan[ui]

Qx = np.real(Qxt)
Qf = np.real(Qft)

print(f"U = {U:.6f}, Qx = {Qx:.6f}, Qf = {Qf:.6f}, Z = {Z:.6f}, lambda = {lmbda:.6f}")

# ----------------------------------------------------------------------
# Precompute V(q), Ep(q), En(q), ak(q) for the whole BZ
# ----------------------------------------------------------------------
def precompute_quantities(kx, ky, kz, U, lmbda, Qx, Qf, d1, d2, d3):
    k_vec = np.stack((kx, ky, kz), axis=-1)
    d1_dot = np.exp(1j * np.tensordot(k_vec, d1, axes=(-1, 0)))
    d2_dot = np.exp(1j * np.tensordot(k_vec, d2, axes=(-1, 0)))
    d3_dot = np.exp(1j * np.tensordot(k_vec, d3, axes=(-1, 0)))
    V = 1.0 + d1_dot + d2_dot + d3_dot
    absV = np.abs(V)
    Ep = np.sqrt(U * (lmbda + Qx * absV))
    En = np.sqrt(U * (lmbda - Qx * absV))
    ak = Qf * absV
    return V, Ep, En, ak

V, Ep, En, ak = precompute_quantities(kx, ky, kz, U, lmbda, Qx, Qf, d1, d2, d3)

# ----------------------------------------------------------------------
# Frequency grid
# ----------------------------------------------------------------------
Nw = 80
aw = 4 * np.pi / 3
W = np.linspace(-aw, aw, Nw)
eta = 0.01          # broadening for non‑local terms
eta1 = 0.05         # broadening for local term
T = 1e-15
beta = 1 / T

# ----------------------------------------------------------------------
# Numba‑accelerated BZ summation (exact translation of MATLAB branches)
# ----------------------------------------------------------------------
@njit(parallel=False, cache=True)   # set parallel=True for extra speed (safe)
def bz_sums(p1, q1, r1, w, eta, beta,
            V, Ep, En, ak, Qf, U_val,
            N1, N2, N3):
    """
    Compute Σ1, Σ2, Σ3 for a fixed (p1,q1,r1) and frequency w.
    Implements all 8 octant branches exactly as in MATLAB.
    """
    norm = 1.0 / (N1 * N2 * N3)

    sum1 = 0.0 + 0.0j
    sum2 = 0.0 + 0.0j
    sum3 = 0.0 + 0.0j
    count = 0

    # Fermi / Bose helpers with overflow protection
    def fermi_plus(e):
        x = beta * e
        if x > 700.0:
            return 0.0
        elif x < -700.0:
            return 1.0
        else:
            return 1.0 / (1.0 + np.exp(x))

    def fermi_minus(e):
        x = -beta * e
        if x > 700.0:
            return 0.0
        elif x < -700.0:
            return 1.0
        else:
            return 1.0 / (1.0 + np.exp(x))

    def bose_plus(e):
        x = beta * e
        if x > 700.0:
            return 0.0
        elif x < -700.0:
            return -1.0
        else:
            return 1.0 / (np.exp(x) - 1.0)

    def bose_minus(e):
        x = -beta * e
        if x > 700.0:
            return 0.0
        elif x < -700.0:
            return -1.0
        else:
            return 1.0 / (np.exp(x) - 1.0)

    # ------------------------------------------------------------------
    # Loop over all (p,q,r)
    # ------------------------------------------------------------------
    for p in range(N1):
        for q in range(N2):
            for r in range(N3):
                # Skip if V(p,q,r) == 0
                if V[p, q, r] == 0.0 + 0.0j:
                    continue

                # ---------- Branch 1: p<p1 , q<q1 , r<r1 ----------
                if p < p1 and q < q1 and r < r1:
                    sp = p1 - p
                    sq = q1 - q
                    sr = r1 - r
                    if sp >= N1 or sq >= N2 or sr >= N3:
                        continue
                    if V[sp, sq, sr] == 0.0 + 0.0j:
                        continue
                    if abs(En[sp, sq, sr]) < 1e-5:
                        count += 1
                        continue

                    V_pqr = V[p, q, r]
                    V_shift = V[sp, sq, sr]
                    ak_pqr = ak[p, q, r]
                    En_shift = En[sp, sq, sr]
                    Ep_shift = Ep[sp, sq, sr]

                    f_h = fermi_minus(ak_pqr)
                    f_e = fermi_plus(ak_pqr)
                    b_m_En = bose_minus(En_shift)
                    b_p_En = bose_plus(En_shift)
                    b_m_Ep = bose_minus(Ep_shift)
                    b_p_Ep = bose_plus(Ep_shift)

                    # sum1 (diagonal)
                    t1a = 0.5 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + En_shift) * (f_h + b_m_En) -
                        1.0 / (w + 1j * eta - ak_pqr - En_shift) * (f_h + b_p_En)
                    ) * norm
                    t1b = 0.5 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + Ep_shift) * (f_h + b_m_Ep) -
                        1.0 / (w + 1j * eta - ak_pqr - Ep_shift) * (f_h + b_p_Ep)
                    ) * norm
                    t1c = 0.5 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + En_shift) * (f_e + b_m_En) -
                        1.0 / (w + 1j * eta + ak_pqr - En_shift) * (f_e + b_p_En)
                    ) * norm
                    t1d = 0.5 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + Ep_shift) * (f_e + b_m_Ep) -
                        1.0 / (w + 1j * eta + ak_pqr - Ep_shift) * (f_e + b_p_Ep)
                    ) * norm
                    sum1 += t1a + t1b + t1c + t1d

                    # sum2 (off‑diagonal)
                    pref2 = (V_shift / np.abs(V_shift)) * Qf * V_pqr / (2.0 * ak_pqr)
                    t2a = pref2 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + En_shift) * (f_e + b_m_En) -
                        1.0 / (w + 1j * eta + ak_pqr - En_shift) * (f_e + b_p_En)
                    ) * norm
                    t2b = -pref2 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + Ep_shift) * (f_e + b_m_Ep) -
                        1.0 / (w + 1j * eta + ak_pqr - Ep_shift) * (f_e + b_p_Ep)
                    ) * norm
                    t2c = -pref2 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + En_shift) * (f_h + b_m_En) -
                        1.0 / (w + 1j * eta - ak_pqr - En_shift) * (f_h + b_p_En)
                    ) * norm
                    t2d = pref2 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + Ep_shift) * (f_h + b_m_Ep) -
                        1.0 / (w + 1j * eta - ak_pqr - Ep_shift) * (f_h + b_p_Ep)
                    ) * norm
                    sum2 += t2a + t2b + t2c + t2d

                    # sum3 (off‑diagonal conjugate)
                    pref3 = (np.conj(V_shift) / np.abs(V_shift)) * Qf * np.conj(V_pqr) / (2.0 * ak_pqr)
                    t3a = pref3 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + En_shift) * (f_e + b_m_En) -
                        1.0 / (w + 1j * eta + ak_pqr - En_shift) * (f_e + b_p_En)
                    ) * norm
                    t3b = -pref3 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + Ep_shift) * (f_e + b_m_Ep) -
                        1.0 / (w + 1j * eta + ak_pqr - Ep_shift) * (f_e + b_p_Ep)
                    ) * norm
                    t3c = -pref3 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + En_shift) * (f_h + b_m_En) -
                        1.0 / (w + 1j * eta - ak_pqr - En_shift) * (f_h + b_p_En)
                    ) * norm
                    t3d = pref3 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + Ep_shift) * (f_h + b_m_Ep) -
                        1.0 / (w + 1j * eta - ak_pqr - Ep_shift) * (f_h + b_p_Ep)
                    ) * norm
                    sum3 += t3a + t3b + t3c + t3d

                # ---------- Branch 2: p<p1 , q>=q1 , r<r1 ----------
                elif p < p1 and q >= q1 and r < r1:
                    sp = p1 - p
                    sq = q1 - q + N2
                    sr = r1 - r
                    if sp >= N1 or sq >= N2 or sr >= N3:
                        continue
                    if V[sp, sq, sr] == 0.0 + 0.0j:
                        continue
                    if abs(En[sp, sq, sr]) < 1e-5:
                        count += 1
                        continue

                    V_pqr = V[p, q, r]
                    V_shift = V[sp, sq, sr]
                    ak_pqr = ak[p, q, r]
                    En_shift = En[sp, sq, sr]
                    Ep_shift = Ep[sp, sq, sr]

                    f_h = fermi_minus(ak_pqr)
                    f_e = fermi_plus(ak_pqr)
                    b_m_En = bose_minus(En_shift)
                    b_p_En = bose_plus(En_shift)
                    b_m_Ep = bose_minus(Ep_shift)
                    b_p_Ep = bose_plus(Ep_shift)

                    # sum1 (same pattern, branch 2)
                    t1a = 0.5 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + En_shift) * (f_h + b_m_En) -
                        1.0 / (w + 1j * eta - ak_pqr - En_shift) * (f_h + b_p_En)
                    ) * norm
                    t1b = 0.5 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + Ep_shift) * (f_h + b_m_Ep) -
                        1.0 / (w + 1j * eta - ak_pqr - Ep_shift) * (f_h + b_p_Ep)
                    ) * norm
                    t1c = 0.5 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + En_shift) * (f_e + b_m_En) -
                        1.0 / (w + 1j * eta + ak_pqr - En_shift) * (f_e + b_p_En)
                    ) * norm
                    t1d = 0.5 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + Ep_shift) * (f_e + b_m_Ep) -
                        1.0 / (w + 1j * eta + ak_pqr - Ep_shift) * (f_e + b_p_Ep)
                    ) * norm
                    sum1 += t1a + t1b + t1c + t1d

                    # sum2
                    pref2 = (V_shift / np.abs(V_shift)) * Qf * V_pqr / (2.0 * ak_pqr)
                    t2a = pref2 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + En_shift) * (f_e + b_m_En) -
                        1.0 / (w + 1j * eta + ak_pqr - En_shift) * (f_e + b_p_En)
                    ) * norm
                    t2b = -pref2 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + Ep_shift) * (f_e + b_m_Ep) -
                        1.0 / (w + 1j * eta + ak_pqr - Ep_shift) * (f_e + b_p_Ep)
                    ) * norm
                    t2c = -pref2 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + En_shift) * (f_h + b_m_En) -
                        1.0 / (w + 1j * eta - ak_pqr - En_shift) * (f_h + b_p_En)
                    ) * norm
                    t2d = pref2 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + Ep_shift) * (f_h + b_m_Ep) -
                        1.0 / (w + 1j * eta - ak_pqr - Ep_shift) * (f_h + b_p_Ep)
                    ) * norm
                    sum2 += t2a + t2b + t2c + t2d

                    # sum3
                    pref3 = (np.conj(V_shift) / np.abs(V_shift)) * Qf * np.conj(V_pqr) / (2.0 * ak_pqr)
                    t3a = pref3 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + En_shift) * (f_e + b_m_En) -
                        1.0 / (w + 1j * eta + ak_pqr - En_shift) * (f_e + b_p_En)
                    ) * norm
                    t3b = -pref3 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + Ep_shift) * (f_e + b_m_Ep) -
                        1.0 / (w + 1j * eta + ak_pqr - Ep_shift) * (f_e + b_p_Ep)
                    ) * norm
                    t3c = -pref3 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + En_shift) * (f_h + b_m_En) -
                        1.0 / (w + 1j * eta - ak_pqr - En_shift) * (f_h + b_p_En)
                    ) * norm
                    t3d = pref3 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + Ep_shift) * (f_h + b_m_Ep) -
                        1.0 / (w + 1j * eta - ak_pqr - Ep_shift) * (f_h + b_p_Ep)
                    ) * norm
                    sum3 += t3a + t3b + t3c + t3d

                # ---------- Branch 3: p>=p1 , q<q1 , r<r1 ----------
                elif p >= p1 and q < q1 and r < r1:
                    sp = p1 - p + N1
                    sq = q1 - q
                    sr = r1 - r
                    if sp >= N1 or sq >= N2 or sr >= N3:
                        continue
                    if V[sp, sq, sr] == 0.0 + 0.0j:
                        continue
                    if abs(En[sp, sq, sr]) < 1e-5:
                        count += 1
                        continue

                    V_pqr = V[p, q, r]
                    V_shift = V[sp, sq, sr]
                    ak_pqr = ak[p, q, r]
                    En_shift = En[sp, sq, sr]
                    Ep_shift = Ep[sp, sq, sr]

                    f_h = fermi_minus(ak_pqr)
                    f_e = fermi_plus(ak_pqr)
                    b_m_En = bose_minus(En_shift)
                    b_p_En = bose_plus(En_shift)
                    b_m_Ep = bose_minus(Ep_shift)
                    b_p_Ep = bose_plus(Ep_shift)

                    # sum1 (branch 3)
                    t1a = 0.5 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + En_shift) * (f_h + b_m_En) -
                        1.0 / (w + 1j * eta - ak_pqr - En_shift) * (f_h + b_p_En)
                    ) * norm
                    t1b = 0.5 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + Ep_shift) * (f_h + b_m_Ep) -
                        1.0 / (w + 1j * eta - ak_pqr - Ep_shift) * (f_h + b_p_Ep)
                    ) * norm
                    t1c = 0.5 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + En_shift) * (f_e + b_m_En) -
                        1.0 / (w + 1j * eta + ak_pqr - En_shift) * (f_e + b_p_En)
                    ) * norm
                    t1d = 0.5 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + Ep_shift) * (f_e + b_m_Ep) -
                        1.0 / (w + 1j * eta + ak_pqr - Ep_shift) * (f_e + b_p_Ep)
                    ) * norm
                    sum1 += t1a + t1b + t1c + t1d

                    # sum2
                    pref2 = (V_shift / np.abs(V_shift)) * Qf * V_pqr / (2.0 * ak_pqr)
                    t2a = pref2 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + En_shift) * (f_e + b_m_En) -
                        1.0 / (w + 1j * eta + ak_pqr - En_shift) * (f_e + b_p_En)
                    ) * norm
                    t2b = -pref2 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + Ep_shift) * (f_e + b_m_Ep) -
                        1.0 / (w + 1j * eta + ak_pqr - Ep_shift) * (f_e + b_p_Ep)
                    ) * norm
                    t2c = -pref2 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + En_shift) * (f_h + b_m_En) -
                        1.0 / (w + 1j * eta - ak_pqr - En_shift) * (f_h + b_p_En)
                    ) * norm
                    t2d = pref2 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + Ep_shift) * (f_h + b_m_Ep) -
                        1.0 / (w + 1j * eta - ak_pqr - Ep_shift) * (f_h + b_p_Ep)
                    ) * norm
                    sum2 += t2a + t2b + t2c + t2d

                    # sum3
                    pref3 = (np.conj(V_shift) / np.abs(V_shift)) * Qf * np.conj(V_pqr) / (2.0 * ak_pqr)
                    t3a = pref3 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + En_shift) * (f_e + b_m_En) -
                        1.0 / (w + 1j * eta + ak_pqr - En_shift) * (f_e + b_p_En)
                    ) * norm
                    t3b = -pref3 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + Ep_shift) * (f_e + b_m_Ep) -
                        1.0 / (w + 1j * eta + ak_pqr - Ep_shift) * (f_e + b_p_Ep)
                    ) * norm
                    t3c = -pref3 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + En_shift) * (f_h + b_m_En) -
                        1.0 / (w + 1j * eta - ak_pqr - En_shift) * (f_h + b_p_En)
                    ) * norm
                    t3d = pref3 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + Ep_shift) * (f_h + b_m_Ep) -
                        1.0 / (w + 1j * eta - ak_pqr - Ep_shift) * (f_h + b_p_Ep)
                    ) * norm
                    sum3 += t3a + t3b + t3c + t3d

                # ---------- Branch 4: p>=p1 , q>=q1 , r<r1 ----------
                elif p >= p1 and q >= q1 and r < r1:
                    sp = p1 - p + N1
                    sq = q1 - q + N2
                    sr = r1 - r
                    if sp >= N1 or sq >= N2 or sr >= N3:
                        continue
                    if V[sp, sq, sr] == 0.0 + 0.0j:
                        continue
                    if abs(En[sp, sq, sr]) < 1e-5:
                        count += 1
                        continue

                    V_pqr = V[p, q, r]
                    V_shift = V[sp, sq, sr]
                    ak_pqr = ak[p, q, r]
                    En_shift = En[sp, sq, sr]
                    Ep_shift = Ep[sp, sq, sr]

                    f_h = fermi_minus(ak_pqr)
                    f_e = fermi_plus(ak_pqr)
                    b_m_En = bose_minus(En_shift)
                    b_p_En = bose_plus(En_shift)
                    b_m_Ep = bose_minus(Ep_shift)
                    b_p_Ep = bose_plus(Ep_shift)

                    # sum1 (branch 4)
                    t1a = 0.5 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + En_shift) * (f_h + b_m_En) -
                        1.0 / (w + 1j * eta - ak_pqr - En_shift) * (f_h + b_p_En)
                    ) * norm
                    t1b = 0.5 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + Ep_shift) * (f_h + b_m_Ep) -
                        1.0 / (w + 1j * eta - ak_pqr - Ep_shift) * (f_h + b_p_Ep)
                    ) * norm
                    t1c = 0.5 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + En_shift) * (f_e + b_m_En) -
                        1.0 / (w + 1j * eta + ak_pqr - En_shift) * (f_e + b_p_En)
                    ) * norm
                    t1d = 0.5 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + Ep_shift) * (f_e + b_m_Ep) -
                        1.0 / (w + 1j * eta + ak_pqr - Ep_shift) * (f_e + b_p_Ep)
                    ) * norm
                    sum1 += t1a + t1b + t1c + t1d

                    # sum2
                    pref2 = (V_shift / np.abs(V_shift)) * Qf * V_pqr / (2.0 * ak_pqr)
                    t2a = pref2 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + En_shift) * (f_e + b_m_En) -
                        1.0 / (w + 1j * eta + ak_pqr - En_shift) * (f_e + b_p_En)
                    ) * norm
                    t2b = -pref2 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + Ep_shift) * (f_e + b_m_Ep) -
                        1.0 / (w + 1j * eta + ak_pqr - Ep_shift) * (f_e + b_p_Ep)
                    ) * norm
                    t2c = -pref2 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + En_shift) * (f_h + b_m_En) -
                        1.0 / (w + 1j * eta - ak_pqr - En_shift) * (f_h + b_p_En)
                    ) * norm
                    t2d = pref2 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + Ep_shift) * (f_h + b_m_Ep) -
                        1.0 / (w + 1j * eta - ak_pqr - Ep_shift) * (f_h + b_p_Ep)
                    ) * norm
                    sum2 += t2a + t2b + t2c + t2d

                    # sum3
                    pref3 = (np.conj(V_shift) / np.abs(V_shift)) * Qf * np.conj(V_pqr) / (2.0 * ak_pqr)
                    t3a = pref3 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + En_shift) * (f_e + b_m_En) -
                        1.0 / (w + 1j * eta + ak_pqr - En_shift) * (f_e + b_p_En)
                    ) * norm
                    t3b = -pref3 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + Ep_shift) * (f_e + b_m_Ep) -
                        1.0 / (w + 1j * eta + ak_pqr - Ep_shift) * (f_e + b_p_Ep)
                    ) * norm
                    t3c = -pref3 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + En_shift) * (f_h + b_m_En) -
                        1.0 / (w + 1j * eta - ak_pqr - En_shift) * (f_h + b_p_En)
                    ) * norm
                    t3d = pref3 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + Ep_shift) * (f_h + b_m_Ep) -
                        1.0 / (w + 1j * eta - ak_pqr - Ep_shift) * (f_h + b_p_Ep)
                    ) * norm
                    sum3 += t3a + t3b + t3c + t3d

                # ---------- Branch 5: p<p1 , q<q1 , r>=r1 ----------
                elif p < p1 and q < q1 and r >= r1:
                    sp = p1 - p
                    sq = q1 - q
                    sr = r1 - r + N3
                    if sp >= N1 or sq >= N2 or sr >= N3:
                        continue
                    if V[sp, sq, sr] == 0.0 + 0.0j:
                        continue
                    if abs(En[sp, sq, sr]) < 1e-5:
                        count += 1
                        continue

                    V_pqr = V[p, q, r]
                    V_shift = V[sp, sq, sr]
                    ak_pqr = ak[p, q, r]
                    En_shift = En[sp, sq, sr]
                    Ep_shift = Ep[sp, sq, sr]

                    f_h = fermi_minus(ak_pqr)
                    f_e = fermi_plus(ak_pqr)
                    b_m_En = bose_minus(En_shift)
                    b_p_En = bose_plus(En_shift)
                    b_m_Ep = bose_minus(Ep_shift)
                    b_p_Ep = bose_plus(Ep_shift)

                    # sum1 (branch 5)
                    t1a = 0.5 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + En_shift) * (f_h + b_m_En) -
                        1.0 / (w + 1j * eta - ak_pqr - En_shift) * (f_h + b_p_En)
                    ) * norm
                    t1b = 0.5 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + Ep_shift) * (f_h + b_m_Ep) -
                        1.0 / (w + 1j * eta - ak_pqr - Ep_shift) * (f_h + b_p_Ep)
                    ) * norm
                    t1c = 0.5 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + En_shift) * (f_e + b_m_En) -
                        1.0 / (w + 1j * eta + ak_pqr - En_shift) * (f_e + b_p_En)
                    ) * norm
                    t1d = 0.5 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + Ep_shift) * (f_e + b_m_Ep) -
                        1.0 / (w + 1j * eta + ak_pqr - Ep_shift) * (f_e + b_p_Ep)
                    ) * norm
                    sum1 += t1a + t1b + t1c + t1d

                    # sum2
                    pref2 = (V_shift / np.abs(V_shift)) * Qf * V_pqr / (2.0 * ak_pqr)
                    t2a = pref2 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + En_shift) * (f_e + b_m_En) -
                        1.0 / (w + 1j * eta + ak_pqr - En_shift) * (f_e + b_p_En)
                    ) * norm
                    t2b = -pref2 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + Ep_shift) * (f_e + b_m_Ep) -
                        1.0 / (w + 1j * eta + ak_pqr - Ep_shift) * (f_e + b_p_Ep)
                    ) * norm
                    t2c = -pref2 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + En_shift) * (f_h + b_m_En) -
                        1.0 / (w + 1j * eta - ak_pqr - En_shift) * (f_h + b_p_En)
                    ) * norm
                    t2d = pref2 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + Ep_shift) * (f_h + b_m_Ep) -
                        1.0 / (w + 1j * eta - ak_pqr - Ep_shift) * (f_h + b_p_Ep)
                    ) * norm
                    sum2 += t2a + t2b + t2c + t2d

                    # sum3
                    pref3 = (np.conj(V_shift) / np.abs(V_shift)) * Qf * np.conj(V_pqr) / (2.0 * ak_pqr)
                    t3a = pref3 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + En_shift) * (f_e + b_m_En) -
                        1.0 / (w + 1j * eta + ak_pqr - En_shift) * (f_e + b_p_En)
                    ) * norm
                    t3b = -pref3 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + Ep_shift) * (f_e + b_m_Ep) -
                        1.0 / (w + 1j * eta + ak_pqr - Ep_shift) * (f_e + b_p_Ep)
                    ) * norm
                    t3c = -pref3 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + En_shift) * (f_h + b_m_En) -
                        1.0 / (w + 1j * eta - ak_pqr - En_shift) * (f_h + b_p_En)
                    ) * norm
                    t3d = pref3 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + Ep_shift) * (f_h + b_m_Ep) -
                        1.0 / (w + 1j * eta - ak_pqr - Ep_shift) * (f_h + b_p_Ep)
                    ) * norm
                    sum3 += t3a + t3b + t3c + t3d

                # ---------- Branch 6: p<p1 , q>=q1 , r>=r1 ----------
                elif p < p1 and q >= q1 and r >= r1:
                    sp = p1 - p
                    sq = q1 - q + N2
                    sr = r1 - r + N3
                    if sp >= N1 or sq >= N2 or sr >= N3:
                        continue
                    if V[sp, sq, sr] == 0.0 + 0.0j:
                        continue
                    if abs(En[sp, sq, sr]) < 1e-5:
                        count += 1
                        continue

                    V_pqr = V[p, q, r]
                    V_shift = V[sp, sq, sr]
                    ak_pqr = ak[p, q, r]
                    En_shift = En[sp, sq, sr]
                    Ep_shift = Ep[sp, sq, sr]

                    f_h = fermi_minus(ak_pqr)
                    f_e = fermi_plus(ak_pqr)
                    b_m_En = bose_minus(En_shift)
                    b_p_En = bose_plus(En_shift)
                    b_m_Ep = bose_minus(Ep_shift)
                    b_p_Ep = bose_plus(Ep_shift)

                    # sum1 (branch 6)
                    t1a = 0.5 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + En_shift) * (f_h + b_m_En) -
                        1.0 / (w + 1j * eta - ak_pqr - En_shift) * (f_h + b_p_En)
                    ) * norm
                    t1b = 0.5 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + Ep_shift) * (f_h + b_m_Ep) -
                        1.0 / (w + 1j * eta - ak_pqr - Ep_shift) * (f_h + b_p_Ep)
                    ) * norm
                    t1c = 0.5 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + En_shift) * (f_e + b_m_En) -
                        1.0 / (w + 1j * eta + ak_pqr - En_shift) * (f_e + b_p_En)
                    ) * norm
                    t1d = 0.5 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + Ep_shift) * (f_e + b_m_Ep) -
                        1.0 / (w + 1j * eta + ak_pqr - Ep_shift) * (f_e + b_p_Ep)
                    ) * norm
                    sum1 += t1a + t1b + t1c + t1d

                    # sum2
                    pref2 = (V_shift / np.abs(V_shift)) * Qf * V_pqr / (2.0 * ak_pqr)
                    t2a = pref2 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + En_shift) * (f_e + b_m_En) -
                        1.0 / (w + 1j * eta + ak_pqr - En_shift) * (f_e + b_p_En)
                    ) * norm
                    t2b = -pref2 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + Ep_shift) * (f_e + b_m_Ep) -
                        1.0 / (w + 1j * eta + ak_pqr - Ep_shift) * (f_e + b_p_Ep)
                    ) * norm
                    t2c = -pref2 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + En_shift) * (f_h + b_m_En) -
                        1.0 / (w + 1j * eta - ak_pqr - En_shift) * (f_h + b_p_En)
                    ) * norm
                    t2d = pref2 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + Ep_shift) * (f_h + b_m_Ep) -
                        1.0 / (w + 1j * eta - ak_pqr - Ep_shift) * (f_h + b_p_Ep)
                    ) * norm
                    sum2 += t2a + t2b + t2c + t2d

                    # sum3
                    pref3 = (np.conj(V_shift) / np.abs(V_shift)) * Qf * np.conj(V_pqr) / (2.0 * ak_pqr)
                    t3a = pref3 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + En_shift) * (f_e + b_m_En) -
                        1.0 / (w + 1j * eta + ak_pqr - En_shift) * (f_e + b_p_En)
                    ) * norm
                    t3b = -pref3 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + Ep_shift) * (f_e + b_m_Ep) -
                        1.0 / (w + 1j * eta + ak_pqr - Ep_shift) * (f_e + b_p_Ep)
                    ) * norm
                    t3c = -pref3 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + En_shift) * (f_h + b_m_En) -
                        1.0 / (w + 1j * eta - ak_pqr - En_shift) * (f_h + b_p_En)
                    ) * norm
                    t3d = pref3 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + Ep_shift) * (f_h + b_m_Ep) -
                        1.0 / (w + 1j * eta - ak_pqr - Ep_shift) * (f_h + b_p_Ep)
                    ) * norm
                    sum3 += t3a + t3b + t3c + t3d

                # ---------- Branch 7: p>=p1 , q<q1 , r>=r1 ----------
                elif p >= p1 and q < q1 and r >= r1:
                    sp = p1 - p + N1
                    sq = q1 - q
                    sr = r1 - r + N3
                    if sp >= N1 or sq >= N2 or sr >= N3:
                        continue
                    if V[sp, sq, sr] == 0.0 + 0.0j:
                        continue
                    if abs(En[sp, sq, sr]) < 1e-5:
                        count += 1
                        continue

                    V_pqr = V[p, q, r]
                    V_shift = V[sp, sq, sr]
                    ak_pqr = ak[p, q, r]
                    En_shift = En[sp, sq, sr]
                    Ep_shift = Ep[sp, sq, sr]

                    f_h = fermi_minus(ak_pqr)
                    f_e = fermi_plus(ak_pqr)
                    b_m_En = bose_minus(En_shift)
                    b_p_En = bose_plus(En_shift)
                    b_m_Ep = bose_minus(Ep_shift)
                    b_p_Ep = bose_plus(Ep_shift)

                    # sum1 (branch 7)
                    t1a = 0.5 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + En_shift) * (f_h + b_m_En) -
                        1.0 / (w + 1j * eta - ak_pqr - En_shift) * (f_h + b_p_En)
                    ) * norm
                    t1b = 0.5 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + Ep_shift) * (f_h + b_m_Ep) -
                        1.0 / (w + 1j * eta - ak_pqr - Ep_shift) * (f_h + b_p_Ep)
                    ) * norm
                    t1c = 0.5 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + En_shift) * (f_e + b_m_En) -
                        1.0 / (w + 1j * eta + ak_pqr - En_shift) * (f_e + b_p_En)
                    ) * norm
                    t1d = 0.5 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + Ep_shift) * (f_e + b_m_Ep) -
                        1.0 / (w + 1j * eta + ak_pqr - Ep_shift) * (f_e + b_p_Ep)
                    ) * norm
                    sum1 += t1a + t1b + t1c + t1d

                    # sum2
                    pref2 = (V_shift / np.abs(V_shift)) * Qf * V_pqr / (2.0 * ak_pqr)
                    t2a = pref2 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + En_shift) * (f_e + b_m_En) -
                        1.0 / (w + 1j * eta + ak_pqr - En_shift) * (f_e + b_p_En)
                    ) * norm
                    t2b = -pref2 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + Ep_shift) * (f_e + b_m_Ep) -
                        1.0 / (w + 1j * eta + ak_pqr - Ep_shift) * (f_e + b_p_Ep)
                    ) * norm
                    t2c = -pref2 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + En_shift) * (f_h + b_m_En) -
                        1.0 / (w + 1j * eta - ak_pqr - En_shift) * (f_h + b_p_En)
                    ) * norm
                    t2d = pref2 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + Ep_shift) * (f_h + b_m_Ep) -
                        1.0 / (w + 1j * eta - ak_pqr - Ep_shift) * (f_h + b_p_Ep)
                    ) * norm
                    sum2 += t2a + t2b + t2c + t2d

                    # sum3
                    pref3 = (np.conj(V_shift) / np.abs(V_shift)) * Qf * np.conj(V_pqr) / (2.0 * ak_pqr)
                    t3a = pref3 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + En_shift) * (f_e + b_m_En) -
                        1.0 / (w + 1j * eta + ak_pqr - En_shift) * (f_e + b_p_En)
                    ) * norm
                    t3b = -pref3 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + Ep_shift) * (f_e + b_m_Ep) -
                        1.0 / (w + 1j * eta + ak_pqr - Ep_shift) * (f_e + b_p_Ep)
                    ) * norm
                    t3c = -pref3 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + En_shift) * (f_h + b_m_En) -
                        1.0 / (w + 1j * eta - ak_pqr - En_shift) * (f_h + b_p_En)
                    ) * norm
                    t3d = pref3 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + Ep_shift) * (f_h + b_m_Ep) -
                        1.0 / (w + 1j * eta - ak_pqr - Ep_shift) * (f_h + b_p_Ep)
                    ) * norm
                    sum3 += t3a + t3b + t3c + t3d

                # ---------- Branch 8: p>=p1 , q>=q1 , r>=r1 ----------
                elif p >= p1 and q >= q1 and r >= r1:
                    sp = p1 - p + N1
                    sq = q1 - q + N2
                    sr = r1 - r + N3
                    if sp >= N1 or sq >= N2 or sr >= N3:
                        continue
                    if V[sp, sq, sr] == 0.0 + 0.0j:
                        continue
                    if abs(En[sp, sq, sr]) < 1e-5:
                        count += 1
                        continue

                    V_pqr = V[p, q, r]
                    V_shift = V[sp, sq, sr]
                    ak_pqr = ak[p, q, r]
                    En_shift = En[sp, sq, sr]
                    Ep_shift = Ep[sp, q, r]  # careful: original code uses Ep[sp, sq, sr]? Actually check: in branch 8, they use Ep(p1-p+N1,q1-q+N2,r1-r+N3). Yes, that's correct.
                    # Actually the original code uses Ep(p1-p+N1,q1-q+N2,r1-r+N3). So we need Ep[sp, sq, sr] (consistent with above). I'll fix: 
                    # It should be Ep_shift = Ep[sp, sq, sr] (already defined as such)
                    # So I'll keep it as Ep_shift = Ep[sp, sq, sr] but the variable was overwritten incorrectly. Let's ensure we use the correct indices.
                    # I'll reassign correctly:
                    En_shift = En[sp, sq, sr]
                    Ep_shift = Ep[sp, sq, sr]

                    f_h = fermi_minus(ak_pqr)
                    f_e = fermi_plus(ak_pqr)
                    b_m_En = bose_minus(En_shift)
                    b_p_En = bose_plus(En_shift)
                    b_m_Ep = bose_minus(Ep_shift)
                    b_p_Ep = bose_plus(Ep_shift)

                    # sum1 (branch 8)
                    t1a = 0.5 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + En_shift) * (f_h + b_m_En) -
                        1.0 / (w + 1j * eta - ak_pqr - En_shift) * (f_h + b_p_En)
                    ) * norm
                    t1b = 0.5 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + Ep_shift) * (f_h + b_m_Ep) -
                        1.0 / (w + 1j * eta - ak_pqr - Ep_shift) * (f_h + b_p_Ep)
                    ) * norm
                    t1c = 0.5 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + En_shift) * (f_e + b_m_En) -
                        1.0 / (w + 1j * eta + ak_pqr - En_shift) * (f_e + b_p_En)
                    ) * norm
                    t1d = 0.5 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + Ep_shift) * (f_e + b_m_Ep) -
                        1.0 / (w + 1j * eta + ak_pqr - Ep_shift) * (f_e + b_p_Ep)
                    ) * norm
                    sum1 += t1a + t1b + t1c + t1d

                    # sum2
                    pref2 = (V_shift / np.abs(V_shift)) * Qf * V_pqr / (2.0 * ak_pqr)
                    t2a = pref2 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + En_shift) * (f_e + b_m_En) -
                        1.0 / (w + 1j * eta + ak_pqr - En_shift) * (f_e + b_p_En)
                    ) * norm
                    t2b = -pref2 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + Ep_shift) * (f_e + b_m_Ep) -
                        1.0 / (w + 1j * eta + ak_pqr - Ep_shift) * (f_e + b_p_Ep)
                    ) * norm
                    t2c = -pref2 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + En_shift) * (f_h + b_m_En) -
                        1.0 / (w + 1j * eta - ak_pqr - En_shift) * (f_h + b_p_En)
                    ) * norm
                    t2d = pref2 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + Ep_shift) * (f_h + b_m_Ep) -
                        1.0 / (w + 1j * eta - ak_pqr - Ep_shift) * (f_h + b_p_Ep)
                    ) * norm
                    sum2 += t2a + t2b + t2c + t2d

                    # sum3
                    pref3 = (np.conj(V_shift) / np.abs(V_shift)) * Qf * np.conj(V_pqr) / (2.0 * ak_pqr)
                    t3a = pref3 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + En_shift) * (f_e + b_m_En) -
                        1.0 / (w + 1j * eta + ak_pqr - En_shift) * (f_e + b_p_En)
                    ) * norm
                    t3b = -pref3 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta + ak_pqr + Ep_shift) * (f_e + b_m_Ep) -
                        1.0 / (w + 1j * eta + ak_pqr - Ep_shift) * (f_e + b_p_Ep)
                    ) * norm
                    t3c = -pref3 * (U_val / (2 * En_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + En_shift) * (f_h + b_m_En) -
                        1.0 / (w + 1j * eta - ak_pqr - En_shift) * (f_h + b_p_En)
                    ) * norm
                    t3d = pref3 * (U_val / (2 * Ep_shift)) * (
                        1.0 / (w + 1j * eta - ak_pqr + Ep_shift) * (f_h + b_m_Ep) -
                        1.0 / (w + 1j * eta - ak_pqr - Ep_shift) * (f_h + b_p_Ep)
                    ) * norm
                    sum3 += t3a + t3b + t3c + t3d

    return sum1, sum2, sum3, count

# ----------------------------------------------------------------------
# Main calculation: only the first k‑point of the path (as in original)
# ----------------------------------------------------------------------
# Only one k‑point (f=1) to match MATLAB's `for f=1:1;`
f_idx = 0   # first point in the path
p1 = P1[f_idx]
q1 = Q1[f_idx]
r1 = R1[f_idx]

print(f"\nComputing for k-point: p1={p1}, q1={q1}, r1={r1}")
print(f"k = [{kx[p1,q1,r1]:.6f}, {ky[p1,q1,r1]:.6f}, {kz[p1,q1,r1]:.6f}] (2π units)")

# Storage for results
Ga = np.zeros(Nw, dtype=np.complex128)          # determinant
Ga2 = np.zeros(Nw, dtype=np.float64)            # -1/π Im(Tr G)
Green_mat = np.zeros((Nw, 2, 2), dtype=np.complex128)  # G matrix for each ω

# Frequency loop
for iw, w in enumerate(tqdm(W, desc="Frequencies")):
    # BZ summation (Numba accelerated)
    sum1, sum2, sum3, _ = bz_sums(p1, q1, r1, w, eta, beta,
                                  V, Ep, En, ak, Qf, U,
                                  N1, N2, N3)

    # Local term (same as MATLAB: Z * ... /2 )
    local_diag = Z * (1.0 / (w + 1j*eta1 - ak[p1,q1,r1]) +
                      1.0 / (w + 1j*eta1 + ak[p1,q1,r1])) / 2.0
    local_off = Z * Qf * V[p1,q1,r1] / ak[p1,q1,r1] * \
                (1.0 / (w + 1j*eta1 - ak[p1,q1,r1]) -
                 1.0 / (w + 1j*eta1 + ak[p1,q1,r1])) / 2.0
    local_off_conj = Z * Qf * np.conj(V[p1,q1,r1]) / ak[p1,q1,r1] * \
                     (1.0 / (w + 1j*eta1 - ak[p1,q1,r1]) -
                      1.0 / (w + 1j*eta1 + ak[p1,q1,r1])) / 2.0

    # Build 2x2 Green's function matrix
    A = np.zeros((2, 2), dtype=np.complex128)
    A[0, 0] = sum1 / 2.0 + local_diag
    A[0, 1] = sum2 / 2.0 + local_off
    A[1, 0] = sum3 / 2.0 + local_off_conj
    A[1, 1] = sum1 / 2.0 + local_diag   # diagonal elements are equal

    # Store
    Green_mat[iw, :, :] = A
    Ga[iw] = np.linalg.det(A)
    Ga2[iw] = -1.0 / np.pi * np.imag(A[0,0] + A[1,1])

    # (Optional) print like MATLAB
    if iw % 10 == 0:
        print(f"ω = {w:.4f}, -1/π Im TrG = {Ga2[iw]:.6f}")

# ----------------------------------------------------------------------
# Save results to .mat files (compatible with MATLAB)
# ----------------------------------------------------------------------
sio.savemat('GreenDet41.mat', {'Ga': Ga.reshape(1,1,Nw)})   # shape (1,1,Nw)
sio.savemat('Green41.mat', {'Green': Green_mat.reshape(1,1,Nw,2,2)})

print("\nResults saved to GreenDet41.mat and Green41.mat")

# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(W, np.abs(Ga), 'b-', linewidth=1.5)
plt.xlabel(r'$\omega$')
plt.ylabel(r'$|\det G(\mathbf{k},\omega)|$')
plt.title('Green’s function determinant')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(W, Ga2, 'r-', linewidth=1.5)
plt.xlabel(r'$\omega$')
plt.ylabel(r'$-\frac{1}{\pi}\mathrm{Im}\,\mathrm{Tr}\,G(\mathbf{k},\omega)$')
plt.title('Spectral function (trace)')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()