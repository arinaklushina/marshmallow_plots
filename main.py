# -----------------------------------------------------------------------------------
# Boson-star BVP solver and plotting script
#
# Purpose:
#   - Solve a coupled boundary-value problem (BVP) for fields (w, m, gamma, kappa, S, y)
#   - Build derived geometric scalars B, R and their radial derivatives
#   - Produce diagnostic plots, including pairwise scatter panels and profiles
#
# Notes for running:
#   * Requires a LaTeX installation because plt.rcParams['text.usetex'] = True
#   * The labels for B-derivatives use TeX-safe primes: r"$B'$", r"$B''$", r"$B'''$"
#   * Tuning parameters are grouped under "tunables" below
#   * All arrays are evaluated on a geometric radial mesh r \in [0.1, RMAX]
#
# READABILITY-ONLY EDITS:
#   The *code itself is unchanged*. Only comments and blank lines were added.
# -----------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from fontTools.misc.roundTools import otRound
from scipy.integrate import solve_bvp

# --------------------------- Matplotlib global settings ----------------------------
plt.rcParams.update({'font.size': 15})
plt.rcParams['font.family'] = 'serif'  # Couldn't find Times New Roman, next best thing.
plt.rcParams['text.usetex'] = True     # Requires LaTeX installed and discoverable

# ---------------------------------- tunables --------------------------------------
RMAX, NPTS = 200.0, 4000         # radial domain max and number of mesh points
KAPPA_INF  = -0.01               # boundary value for kappa at r = RMAX
OMEGA, EPS_B = 1e-3, 1e-8        # frequency-like param; floor to avoid division by 0 in B
A, r0, sigma = 1e-3, 20.0, 2.5   # Gaussian source amplitude, center, width  (change A and re-run)

# -------------------------- Gaussian source and derivatives ------------------------
def f_source_gauss(r, *, A=A, r0=r0, sigma=sigma):
    # Normalized Gaussian profile f(r) = A exp(- (r - r0)^2 / (2 sigma^2))
    r = np.asarray(r); u = (r - r0)/sigma
    return A*np.exp(-0.5*u**2)

def f_source_gauss_prime(r, *, A=A, r0=r0, sigma=sigma):
    # First derivative f'(r)
    f = f_source_gauss(r, A=A, r0=r0, sigma=sigma)
    return f * (-(r - r0)/sigma**2)

def f_source_gauss_second(r, *, A=A, r0=r0, sigma=sigma):
    # Second derivative f''(r)
    f = f_source_gauss(r, A=A, r0=r0, sigma=sigma); u = (r - r0)
    return f * ((u**2)/sigma**4 - 1.0/sigma**2)

# ----------------------- helpers for parameter derivatives -------------------------
# These build (w, m, gamma, kappa) radial derivatives in terms of source f and its derivs.
def first_param_derivs(r, f):
    rs = np.where(r==0, np.finfo(float).eps, r)  # avoid division by zero at r=0
    dw      =  0.5       * rs**3 * f
    dm      = (1.0/12.0) * rs**4 * f
    dgamma  = -0.5       * rs**2 * f
    dkappa  = -(1.0/6.0) * rs    * f
    return dw, dm, dgamma, dkappa

def second_param_derivs(r, f, fp):
    rs = np.where(r==0, np.finfo(float).eps, r)
    d2w      =  1.5       * rs**2 * f + 0.5       * rs**3 * fp
    d2m      = (1.0/3.0)  * rs**3 * f + (1.0/12.0)* rs**4 * fp
    d2gamma  = -rs * f - 0.5 * rs**2 * fp
    d2kappa  = -(1.0/6.0) * (f + rs * fp)
    return d2w, d2m, d2gamma, d2kappa

def third_param_derivs(r, f, fp, fpp):
    rs = np.where(r==0, np.finfo(float).eps, r)
    d3w     = 3.0*rs*f + 3.0*rs**2*fp + 0.5*rs**3*fpp
    d3m     = rs**2*f + (2.0/3.0)*rs**3*fp + (1.0/12.0)*rs**4*fpp
    d3gamma = -f - 2.0*rs*fp - 0.5*rs**2*fpp
    d3kappa = -(1.0/6.0) * (2.0*fp + rs*fpp)
    return d3w, d3m, d3gamma, d3kappa

# ------------------------------ BVP (no p, no partial) -----------------------------
def rhs(r, Y):
    # State vector Y = [w, m, gamma, kappa, S, y]
    w, m, gamma, kappa, S, y = Y
    rs = np.where(r==0, np.finfo(float).eps, r)

    # Effective metric functions:
    B = w - 2.0*m/rs + gamma*rs - kappa*rs**2
    R = 2.0*(w - 1.0)/rs**2 + 6.0*gamma/rs - 12.0*kappa

    # Scalar field equations (regularized by EPS_B via max|B|)
    dS = y / (rs**2 * np.maximum(np.abs(B), EPS_B))
    dy = -rs**2 * ((OMEGA**2 / np.maximum(np.abs(B), EPS_B) - R/6.0) * S - S**3)

    # Source-driven updates for (w, m, gamma, kappa)
    f = f_source_gauss(rs, A=A, r0=r0, sigma=sigma)   # use current A,r0,sigma
    dw, dm, dgamma, dkappa = first_param_derivs(rs, f)
    return np.vstack([dw, dm, dgamma, dkappa, dS, dy])

# Parameters for potential-related quantities (separate from Gaussian sigma)
lam = 1
sigma = 1
R_ =  - 12.0 * KAPPA_INF
Smin_Sq = - 2 * (-R_ / 12 + OMEGA ** 2) / (4 * lam / sigma)

def bc(Y0, YR):
    # Boundary conditions at r=0 (left) and r=RMAX (right)
    w0, m0, g0, k0, S0, y0 = Y0
    wR, mR, gR, kR, SR, yR = YR
    # Enforce: w(0)=1, m(0)=0, gamma(0)=0, kappa(R)=KAPPA_INF, y(0)=0, S(R)=sqrt(max(Smin_Sq,0))
    return np.array([w0-1.0, m0, g0, kR-KAPPA_INF, y0, SR - (np.max([Smin_Sq, 0]))**0.5])

# ------------------------------------- solve ---------------------------------------
GEOM_ON = False
if GEOM_ON:
    r = np.geomspace(0.1, RMAX, NPTS)
else:
    rmin = 0.1
    tail_width = 10.0  # last 10 units dense
    n_tail = int(NPTS * 0.6)  # 60% of nodes in the tail
    n_head = NPTS - n_tail

    r_head = np.linspace(rmin, RMAX - tail_width, n_head, endpoint=False)
    r_tail = np.linspace(RMAX - tail_width, RMAX, n_tail)
    r = np.concatenate([r_head, r_tail])

Y_guess = np.zeros((6, r.size)); Y_guess[0]=1.0; Y_guess[3]=KAPPA_INF; Y_guess[4]=1e3
sol = solve_bvp(rhs, bc, r, Y_guess, tol=1e-12)

# ---------------------------- build B, B', B'', B''' -------------------------------
w, m, gamma, kappa, S, y = sol.sol(r)
rs = np.where(r==0, np.finfo(float).eps, r)

# Gaussian and its derivatives on the solved mesh
f   = f_source_gauss(rs, A=A, r0=r0, sigma=sigma)
fp  = f_source_gauss_prime(rs, A=A, r0=r0, sigma=sigma)
fpp = f_source_gauss_second(rs, A=A, r0=r0, sigma=sigma)

# Parameter derivatives up to third order
dw, dm, dgamma, dkappa     = first_param_derivs(rs, f)
d2w, d2m, d2gamma, d2kappa = second_param_derivs(rs, f, fp)
d3w, d3m, d3gamma, d3kappa = third_param_derivs(rs, f, fp, fpp)

# Effective B and its radial derivatives (assembled via the chain rule)
B  = w - 2.0*m/rs + gamma*rs - kappa*rs**2
Bp = (dw - 2.0*dm/rs + 2.0*m/rs**2 + gamma + rs*dgamma - 2.0*rs*kappa - rs**2*dkappa)
Bpp= (d2w - 2.0*d2m/rs + 4.0*dm/rs**2 - 4.0*m/rs**3 + 2.0*dgamma + rs*d2gamma
      - 2.0*kappa - 4.0*rs*dkappa - rs**2*d2kappa)
Bppp=(d3w - 2.0*d3m/rs + 6.0*d2m/rs**2 - 12.0*dm/rs**3 + 12.0*m/rs**4
      + 3.0*d2gamma + rs*d3gamma - 6.0*dkappa - 6.0*rs*d2kappa - rs**2*d3kappa)

# ----------------------------------- plot: B, B', ... ------------------------------
fig, ax1 = plt.subplots()
ax1.plot(r, B, label="B")
ax1.set_xlabel("r"); ax1.set_ylabel("B")
ax2 = ax1.twinx()
ax2.plot(r, Bp,  label=r"$B'$",  color='purple')
ax2.plot(r, Bpp, label=r"$B''$", linestyle="--")
ax2.plot(r, Bppp,label=r"$B'''$", linestyle=":")

# Merge legends from both axes; add grid and fixed-format ticks
L1,N1 = ax1.get_legend_handles_labels(); L2,N2 = ax2.get_legend_handles_labels()
ax1.legend(L1+L2, N1+N2, loc="best"); ax1.grid(True, alpha=0.3)
ax1.ticklabel_format(axis='y', style='plain', useOffset=False)
ax2.ticklabel_format(axis='y', style='plain', useOffset=False)

# Annotate current Gaussian amplitude A on the plot
ax1.text(
    0.72, 0.18, rf"A = {A:.2e}",
    transform=ax1.transAxes, ha="left", va="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="red")
)

fig.savefig('marshmallow_B.png', dpi=300, bbox_inches='tight')

# -------------------- pack components for pairwise scatter panels ------------------
names = ["w", "m", "gamma", "kappa", "S", "y"]
arrays = [w, m, gamma, kappa, S, y]
origin_vals = []

# Print values of (w, m, gamma, kappa) at the first radial point (proxy for origin)
for i in range(4):
    print(names[i] + " = " + str(arrays[i][0]) + " at the origin")
    origin_vals.append(arrays[i][0])

# Compute R at the "origin" based on those printed values
R_origin = 2*(origin_vals[0] - 1) / r[0]**2 + 6 * origin_vals[2] / r[0] - 12 * origin_vals[3]
print("R at the origin is ", R_origin)

# ----------------------- 6x6 grid of all variable pairs ----------------------------
fig, axes = plt.subplots(6, 6, figsize=(18, 18), sharex=False, sharey=False)

for i in range(6):
    for j in range(6):
        ax = axes[i, j]
        if i == j:
            # Diagonal: just put the variable name
            ax.text(0.5, 0.5, names[i],
                    ha="center", va="center",
                    fontsize=12, weight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
        elif i < j:
            continue
        else:
            # Off-diagonal: scatter (names[i] vs names[j]) colored by r
            xi = arrays[j]   # x-variable
            yi = arrays[i]   # y-variable
            valid = np.isfinite(xi) & np.isfinite(yi)
            sc = ax.scatter(xi[valid], yi[valid], c=r[valid], s=5, cmap="viridis")
            if i == 5:  # bottom row → x-labels
                ax.set_xlabel(names[j])
            if j == 0:  # first column → y-labels
                ax.set_ylabel(names[i])

plt.suptitle("All variable pairs (color = r)", fontsize=16)
plt.tight_layout(rect=[0, 0, 0.95, 0.95])
plt.savefig("marshmallow_all_pairs_6x6.png", dpi=300, bbox_inches="tight")
plt.show()

# ----------------------------- first: differentials --------------------------------
# Finite differences along r for each field: shape (6, N-1)
diff_arrays = np.diff(np.asarray(arrays), axis=1)  # shape (6, N-1)

fig, axes = plt.subplots(3, 2, figsize=(10, 12), sharex=False, sharey=False)
axes = axes.ravel()  # flatten to 1D: 6 Axes

for i, ax in enumerate(axes):
    ax.plot(r[:-1], diff_arrays[i])
    ax.set_ylabel("d" + names[i])
    ax.grid(True, alpha=0.3)

# optional: x-label only on bottom row
for ax in axes[-2:]:
    ax.set_xlabel("r")

plt.tight_layout()
plt.savefig("marshmallow_differentials.png", dpi=300, bbox_inches="tight")


# ---------------------------- second: original arrays ------------------------------
fig, axes = plt.subplots(3, 2, figsize=(10, 12), sharex=False, sharey=False)
axes = axes.ravel()

for i, ax in enumerate(axes):
    ax.plot(r, arrays[i])
    ax.set_ylabel(names[i])
    ax.grid(True, alpha=0.3)

for ax in axes[-2:]:
    ax.set_xlabel("r")

plt.tight_layout()
plt.savefig("marshmallow_parameters.png", dpi=300, bbox_inches="tight")

plt.show()

# -------------------------- potential & curvature diagnostics ----------------------
lam = 1
OMEGA = 0
sig = 1

# Recompute R from solved fields (on full mesh)
R = 2.0*(w - 1.0)/rs**2 + 6.0*gamma/rs - 12.0*kappa

print("R[0] - R_origin = ", R[0] - R_origin)

# Plot R(r)
fig, ax1 = plt.subplots()
ax1.plot(r, R, label="R")
ax1.set_xlabel("r")
ax1.set_ylabel("R")
ax1.grid(True, alpha=0.3)   # add grid with light transparency
ax1.legend()

# --- rightmost local minimum of R(r) ---
valid = np.isfinite(r) & np.isfinite(R)
r_v, R_v = r[valid], R[valid]

dR = np.diff(R_v)
cands = np.where((dR[:-1] < 0) & (dR[1:] > 0))[0] + 1  # minima: slope − then +

if cands.size:
    i_min = cands[-1]  # closest to the right edge
else:
    # fallback: minimum within the last 10% of the domain
    w = max(5, len(R_v)//10)
    base = len(R_v) - w
    i_min = base + np.argmin(R_v[base:])

r_min_right, R_min_right = r_v[i_min], R_v[i_min]
print(f"Rightmost local min: R(r) = {R_min_right:.6g} at r = {r_min_right:.6g}")

# annotate it on the figure (with value shown in legend)
ax1.scatter([r_min_right], [R_min_right], s=40, zorder=5,
            label=rf"rightmost min: $R_{{\min}}={R_min_right:.3g}$ at $r={r_min_right:.3g}$")
ax1.axvline(r_min_right, ls=":", alpha=0.5)
ax1.legend(loc="lower right", fontsize=8, markerscale=0.8, handlelength=1.2, borderpad=0.4)


plt.tight_layout()
fig.savefig("marshmallow_R.png", dpi=300, bbox_inches="tight")
plt.show()

# Potential V(S; r) and helper expressions
V = (-R/12 + OMEGA**2 ) * S**2 + lam / sig * S**4
# dV/dS (symbolic form retained as-is for reference)
# dV = (-(2.0*(dw)/rs**2 - 4.0*(w-1.0) / rs **3 - 6.0*gamma/rs**2 + 6.0*dgamma / rs - 12.0*kappa)/12)*S**2  - R/12*2*S + lam/sig * 4 * S**3
dVdS = (2*(-R/12 + OMEGA**2 ) /(4* lam / sig) + S**2) * S * 4* lam / sig

# Location of potential minimum in S^2, by the quartic form (can be negative => discarded later)
Smin_Sq = - 2*(-R/12 + OMEGA**2 ) /(4* lam / sig)
fig, ax1 = plt.subplots()
ax1.plot(r, Smin_Sq, label=r"$S_{\min}^2(r)$")
ax1.set_xlabel(r"$r$"); ax1.set_ylabel(r"$S_{\min}^2$")
ax1.grid(True, alpha=0.3)   # add grid with light transparency
# --- rightmost minimum of S_min^2(r) ---
valid = np.isfinite(r) & np.isfinite(Smin_Sq)
r_v, s_v = r[valid], Smin_Sq[valid]

# try strict local minima: slope goes − then +
ds = np.diff(s_v)
cands = np.where((ds[:-1] < 0) & (ds[1:] > 0))[0] + 1  # indices of local minima

if cands.size:
    i_min = cands[-1]  # rightmost local minimum
else:
    # fallback: take the minimum within the rightmost 10% of points (at least 5)
    w = max(5, len(s_v)//10)
    base = len(s_v) - w
    i_min = base + np.argmin(s_v[base:])

r_min_right, s_min_right = r_v[i_min], s_v[i_min]
print(f"Rightmost min of Smin^2: {s_min_right:.6g} at r = {r_min_right:.6g}")

# annotate on the figure
ax1.scatter([r_min_right], [s_min_right], s=40, zorder=5,
            label=rf"rightmost min: $S_{{\min}}^2={s_min_right:.3g}$ at $r={r_min_right:.3g}$")
ax1.axvline(r_min_right, ls=":", alpha=0.5)
ax1.legend(loc="lower right", fontsize=9)

fig.savefig("marshmallow_Smin_sq.png", dpi=300, bbox_inches="tight")

# --------------------- V(S) scatter with r-colored gradient ------------------------
valid = np.isfinite(S) & np.isfinite(V) & np.isfinite(r)

fig, ax1 = plt.subplots()
sc = ax1.scatter(S[valid], V[valid], c=r[valid], s=12, cmap="viridis")  # color by r
ax1.set_xlabel(r"$S$")
ax1.set_ylabel(r"$V$")
ax1.grid(True, alpha=0.3)

cbar = fig.colorbar(sc, ax=ax1, label=r"$r$")  # gradient legend
plt.tight_layout()
plt.savefig("marshmallow_Mexican_hat.png", dpi=300, bbox_inches="tight")
plt.show()

# ------------------------------- S(r) profile --------------------------------------
fig, ax1 = plt.subplots()
ax1.plot(r, S, label="S")
ax1.set_xlabel("r"); ax1.set_ylabel("S")
ax1.grid(True, alpha=0.3)   # add grid with light transparency
from scipy.signal import find_peaks

# --- leftmost local maximum of S(r) restricted to [195, r_max] ---
valid = np.isfinite(r) & np.isfinite(S)
r_v_all, s_v_all = r[valid], S[valid]

# restrict to [295, r_max]
i_start = np.searchsorted(r_v_all, 195.0, side="left")
r_v, s_v = r_v_all[i_start:], s_v_all[i_start:]

# first try: local maxima by slope (+ then −)
ds = np.diff(s_v)
cands = np.where((ds[:-1] > 0) & (ds[1:] < 0))[0] + 1  # indices in restricted arrays

if cands.size:
    i0 = cands[0]  # leftmost max within [295, r_max]
else:
    # fallback: global max within [295, r_max]
    i0 = int(np.argmax(s_v))

# parabolic (quadratic) sub-sample refinement around i0
i = int(np.clip(i0, 1, len(s_v) - 2))
y_1, y0, y1 = s_v[i-1], s_v[i], s_v[i+1]
den = (y_1 - 2*y0 + y1)
d = 0.5 * (y_1 - y1) / (den if den != 0 else np.finfo(float).eps)  # vertex offset in index units

r_peak = r_v[i] + d * (r_v[i+1] - r_v[i])
s_peak = y0 - 0.25 * (y_1 - y1) * d

print(f"Leftmost max of S in [295, r_max]: {s_peak:.6g} at r = {r_peak:.6g}")

# annotate like the R-plot snippets
ax1.scatter([r_peak], [s_peak], s=40, zorder=5,
            label=rf"max in $[295,r_{{\max}}]$: $S_{{\max}}={s_peak:.3g}$ at $r={r_peak:.3g}$")
ax1.axvline(r_peak, ls=":", alpha=0.5)
ax1.legend(loc="lower right", fontsize=9)


plt.savefig("marshmallow_S.png", dpi=300, bbox_inches="tight")
plt.show()

# Quick numeric checks on V and S
print("Vmax(S) = ", V[np.argmax(S)])
print("Vmin(S) = ", V[np.argmin(S)])
print("S at Vmin = ", S[np.argmin(V)])

# --- grid of single-variable profiles vs r: w, m, gamma, kappa, S, R, y ---
R_all = 2.0*(w - 1.0)/rs**2 + 6.0*gamma/rs - 12.0*kappa  # compute here in case not yet defined

pairs = [
    (r"$w(r)$", w),
    (r"$m(r)$", m),
    (r"$\gamma(r)$", gamma),
    (r"$\kappa(r)$", kappa),
    (r"$S(r)$", S),
    (r"$R(r)$", R_all),
    (r"$y(r)$", y),
]

nrows, ncols = 4, 2  # 7 panels + 1 empty
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 14), sharex=True)
axes = axes.ravel()

for i, (label, arr) in enumerate(pairs):
    ax = axes[i]
    ax.plot(r, arr)
    ax.set_title(label)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    if i // ncols == nrows - 1:
        ax.set_xlabel(r"$r$")

# hide the unused last panel
for j in range(len(pairs), len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig("marshmallow_fields_grid.png", dpi=300, bbox_inches="tight")
plt.show()

