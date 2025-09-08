# -----------------------------------------------------------------------------------
# Boson-star BVP solver and plotting script  (compactified with r = L * tan(pi x / 2))
# Restores: 6x6 pairwise scatter, differentials panels, annotated S(r) peak, etc.
# -----------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

# --------------------------- Matplotlib global settings ----------------------------
plt.rcParams.update({'font.size': 15})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True

# ---------------------------------- tunables --------------------------------------
RMIN, RMAX, NPTS = 0.1, 800.0, 4000
L = RMAX                        # compactification scale; L ~ RMAX is a good default
KAPPA_INF = -0.00001
OMEGA = 1e-3
EPS_B = 1e-8

# Gaussian source (distinct name from potential's sigma)
A, r0, sigmaG = 1e-9, 40.0, 2.5

# Potential parameters (separate from Gaussian width)
lam = 1.0
sigma_p = 1.0
OMEGA_V = OMEGA                 # keep potential consistent with ODEs (set 0.0 if desired)



# ------------------------- compactification: r = L * tan(pi x / 2) -----------------
def r_of_x(x):
    return L * np.tan(0.5*np.pi*x)

def drdx_of_x(x):
    t = 0.5*np.pi*x
    return (0.5*np.pi*L) * (1.0/np.cos(t))**2

# x-interval mapping exactly to r ∈ [RMIN, RMAX]
x0 = (2.0/np.pi) * np.arctan(RMIN / L)
#x1 = (2.0/np.pi) * np.arctan(L / L)
x1=1
# Optional clustering toward RMAX: p>1 biases nodes near x1 (→ near RMAX)
p = 2.0
s = np.linspace(0.0, 1.0, NPTS)
x = x0 + (x1 - x0) * (1.0 - (1.0 - s)**p)

# -------------------------- Gaussian source and derivatives ------------------------
def f_source_gauss(r, *, A=A, r0=r0, sigma=sigmaG):
    r = np.asarray(r); u = (r - r0)/sigma
    return A*np.exp(-0.5*u**2)

def f_source_gauss_prime(r, *, A=A, r0=r0, sigma=sigmaG):
    f = f_source_gauss(r, A=A, r0=r0, sigma=sigma)
    return f * (-(r - r0)/sigma**2)

def f_source_gauss_second(r, *, A=A, r0=r0, sigma=sigmaG):
    f = f_source_gauss(r, A=A, r0=r0, sigma=sigma); u = (r - r0)
    return f * ((u**2)/sigma**4 - 1.0/sigma**2)

# ----------------------- helpers for parameter derivatives (in r) -------------------
def first_param_derivs(r, f):
    rs = np.where(r==0, np.finfo(float).eps, r)
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

# ------------------------------ BVP in compact coordinate x ------------------------
def rhs(x, Y):
    # Y = [w, m, gamma, kappa, S, y]
    r  = r_of_x(x)
    rs = np.where(r==0, np.finfo(float).eps, r)
    w, m, gamma, kappa, S, y = Y

    B = w - 2.0*m/rs + gamma*rs - kappa*rs**2
    R = 2.0*(w - 1.0)/rs**2 + 6.0*gamma/rs - 12.0*kappa

    dS_dr = y / (rs**2 * np.maximum(np.abs(B), EPS_B))
    dy_dr = -rs**2 * ((OMEGA**2 / np.maximum(np.abs(B), EPS_B) - R/6.0) * S - S**3)

    f  = f_source_gauss(rs)
    dw_dr, dm_dr, dgamma_dr, dkappa_dr = first_param_derivs(rs, f)

    # Chain rule: dY/dx = (dY/dr) * (dr/dx)
    fac = drdx_of_x(x)
    return fac * np.vstack([dw_dr, dm_dr, dgamma_dr, dkappa_dr, dS_dr, dy_dr])

# Asymptotic helper (based on kappa_inf)
R_inf = -12.0 * KAPPA_INF
Smin_Sq_bc = - 2.0 * (-R_inf/12.0) / (4.0 * lam / sigma_p)
S_inf = float(np.sqrt(max(Smin_Sq_bc, 0.0)))

def bc(YL, YR):
    w0, m0, g0, k0, S0, y0 = YL
    wR, mR, gR, kR, SR, yR = YR
    return np.array([
        w0 - (1-6.0*m0*g0)**0.5,               # w(0) = 1
        m0,                     # m(0) = 0
        g0,                     # gamma(0) = 0
        kR - KAPPA_INF,         # kappa(RMAX) = kappa_inf
        y0,                     # y(0) = 0
        SR - S_inf              # S(RMAX) = S_inf
    ])

# ------------------------------------- solve ---------------------------------------
Y_guess = np.zeros((6, x.size))
Y_guess[0] = 1.0
Y_guess[3] = KAPPA_INF
Y_guess[4] = S_inf
Y_guess[5] = 0.0

sol = solve_bvp(rhs, bc, x, Y_guess, tol=1e-14, max_nodes=900000, verbose=2)
if sol.status != 0:
    print("solve_bvp did not converge:", sol.message)

# Evaluate solution and map x→r
r  = r_of_x(x)
w, m, gamma, kappa, S, y = sol.sol(x)
rs = np.where(r==0, np.finfo(float).eps, r)

# ---------------------------- build B, B', B'', B''' -------------------------------
f   = f_source_gauss(rs)
fp  = f_source_gauss_prime(rs)
fpp = f_source_gauss_second(rs)

dw, dm, dgamma, dkappa     = first_param_derivs(rs, f)
d2w, d2m, d2gamma, d2kappa = second_param_derivs(rs, f, fp)
d3w, d3m, d3gamma, d3kappa = third_param_derivs(rs, f, fp, fpp)

B   = w - 2.0*m/rs + gamma*rs - kappa*rs**2
Bp  = (dw - 2.0*dm/rs + 2.0*m/rs**2 + gamma + rs*dgamma - 2.0*rs*kappa - rs**2*dkappa)
Bpp = (d2w - 2.0*d2m/rs + 4.0*dm/rs**2 - 4.0*m/rs**3 + 2.0*dgamma + rs*d2gamma
       - 2.0*kappa - 4.0*rs*dkappa - rs**2*d2kappa)
Bppp= (d3w - 2.0*d3m/rs + 6.0*d2m/rs**2 - 12.0*dm/rs**3 + 12.0*m/rs**4
       + 3.0*d2gamma + rs*d3gamma - 6.0*dkappa - 6.0*rs*d2kappa - rs**2*d3kappa)

# ----------------------------------- helpers ---------------------------------------
def _grid(ax):
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax.set_xlim(RMIN, RMAX)

# ------------------------------ plot: B, B', B'', B''' -----------------------------
fig, ax1 = plt.subplots()
ax1.plot(r, B, label="B")
ax1.set_xlabel("r"); ax1.set_ylabel("B"); _grid(ax1)
ax2 = ax1.twinx()
ax2.plot(r, Bp,  label=r"$B'$", color='purple')
ax2.plot(r, Bpp, label=r"$B''$", linestyle="--")
ax2.plot(r, Bppp,label=r"$B'''$", linestyle=":"); _grid(ax2)
L1,N1 = ax1.get_legend_handles_labels(); L2,N2 = ax2.get_legend_handles_labels()
ax1.legend(L1+L2, N1+N2, loc="best")
fig.savefig("marshmallow_B_tan.png", dpi=300, bbox_inches="tight")

# ---------------------------- single-variable profiles -----------------------------
import matplotlib.ticker as mticker

R_all = 2.0*(w - 1.0)/rs**2 + 6.0*gamma/rs - 12.0*kappa
pairs = [
    (r"$w(r)$", w), (r"$m(r)$", m), (r"$\gamma(r)$", gamma),
    (r"$\kappa(r)$", kappa), (r"$S(r)$", S), (r"$R(r)$", R_all), (r"$y(r)$", y),
]

nrows, ncols = 4, 2
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 14), sharex=True)
axes = axes.ravel()

for i, (lab, arr) in enumerate(pairs):
    ax = axes[i]
    ax.plot(r, arr)
    ax.set_title(lab)
    _grid(ax)
    if i // ncols == nrows - 1:
        ax.set_xlabel("r")

    # Force y-axis into scientific notation
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

# Hide any unused axes
for j in range(len(pairs), len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig("marshmallow_fields_grid_tan.png", dpi=300, bbox_inches="tight")
plt.show()

# ---------------------------- 6x6 pairwise scatter (lower triangle) ----------------
names  = ["w", "m", "gamma", "kappa", "S", "y"]
arrays = [w, m, gamma, kappa, S, y]
fig, axes = plt.subplots(6, 6, figsize=(18, 18), sharex=False, sharey=False)
for i in range(6):
    for j in range(6):
        ax = axes[i, j]
        if i == j:
            ax.text(0.5, 0.5, names[i], ha="center", va="center", fontsize=12, weight="bold")
            ax.set_xticks([]); ax.set_yticks([])
        elif i < j:
            ax.axis("off")
        else:
            xi = arrays[j]; yi = arrays[i]
            valid = np.isfinite(xi) & np.isfinite(yi)
            sc = ax.scatter(xi[valid], yi[valid], c=r[valid], s=5, cmap="viridis")
            if i == 5: ax.set_xlabel(names[j])
            if j == 0: ax.set_ylabel(names[i])
plt.suptitle("All variable pairs (color = r)", fontsize=16)
plt.tight_layout(rect=[0, 0, 0.95, 0.95])
plt.savefig("marshmallow_all_pairs_6x6_tan.png", dpi=300, bbox_inches="tight")
plt.show()

# ----------------------------- finite differences panels ---------------------------
diff_arrays = np.diff(np.asarray(arrays), axis=1)  # shape (6, N-1)
fig, axes = plt.subplots(3, 2, figsize=(10, 12), sharex=False, sharey=False)
axes = axes.ravel()

for i, ax in enumerate(axes):
    ax.plot(r[:-1], diff_arrays[i])
    ax.set_ylabel("d" + names[i])
    _grid(ax)

    # Force y-axis into scientific notation
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

for ax in axes[-2:]:
    ax.set_xlabel("r")

plt.tight_layout()
plt.savefig("marshmallow_differentials_tan.png", dpi=300, bbox_inches="tight")
plt.show()
# -------------------------- potential & curvature diagnostics ----------------------
R = R_all
fig, ax = plt.subplots()
ax.plot(r, R, label="R")
ax.set_xlabel("r")
ax.set_ylabel("R")
_grid(ax)
ax.legend()

dR = np.diff(R)
cands = np.where((dR[:-1] < 0) & (dR[1:] > 0))[0] + 1
i_min = (cands[-1] if cands.size else
         np.argmin(R[-max(5, len(R)//10):]) + len(R) - max(5, len(R)//10))

ax.scatter([r[i_min]], [R[i_min]], s=40, zorder=5,
           label=rf"rightmost min: $R_{{\min}}={R[i_min]:.3g}$ at $r={r[i_min]:.3g}$")
ax.axvline(r[i_min], ls=":", alpha=0.5)

# Force scientific notation on y-axis
ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

ax.legend(loc="lower right", fontsize=8)

plt.tight_layout()
plt.savefig("marshmallow_R_tan.png", dpi=300, bbox_inches="tight")
plt.show()

V = (-R/12 + OMEGA_V**2) * S**2 + lam/sigma_p * S**4
Smin_Sq = - 2*(-R/12 + OMEGA_V**2)/(4*lam/sigma_p)

# --- First plot: S_min^2(r) ---
fig, ax = plt.subplots()
ax.plot(r, Smin_Sq, label=r"$S_{\min}^2(r)$")
ax.set_xlabel("r")
ax.set_ylabel(r"$S_{\min}^2$")
_grid(ax)

# force y-axis into scientific notation
ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

ax.legend()
plt.savefig("marshmallow_Smin_sq_tan.png", dpi=300, bbox_inches="tight")

# --- Second plot: Mexican hat potential ---
valid = np.isfinite(S) & np.isfinite(V) & np.isfinite(r)
fig, ax = plt.subplots()
sc = ax.scatter(S[valid], V[valid], c=r[valid], s=12, cmap="viridis")
ax.set_xlabel(r"$S$")
ax.set_ylabel(r"$V$")
ax.grid(True, alpha=0.3)

# scientific notation for y-axis (and x if desired)
ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

# optional: also scientific for x-axis
# ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
# ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

fig.colorbar(sc, ax=ax, label=r"$r$")
plt.tight_layout()
plt.savefig("marshmallow_Mexican_hat_tan.png", dpi=300, bbox_inches="tight")
plt.show()
# ------------------------------- S(r) profile + robust peak ------------------------
fig, ax = plt.subplots()
ax.plot(r, S, label="S"); ax.set_xlabel("r"); ax.set_ylabel("S"); _grid(ax)

r_left = 195.0
mask = np.isfinite(r) & np.isfinite(S) & (r >= r_left)
r_seg, s_seg = r[mask], S[mask]
if r_seg.size < 3:  # fallback to tail if the window is too small
    Ntail = min(200, r.size)
    r_seg, s_seg = r[-Ntail:], S[-Ntail:]

if r_seg.size >= 3:
    ds = np.diff(s_seg)
    cands = np.where((ds[:-1] > 0) & (ds[1:] < 0))[0] + 1
    i0 = cands[0] if cands.size else int(np.argmax(s_seg))
    i  = int(np.clip(i0, 1, len(s_seg)-2))
    y_1, y0, y1 = s_seg[i-1], s_seg[i], s_seg[i+1]
    den = (y_1 - 2*y0 + y1)
    d = 0.5 * (y_1 - y1) / (den if den != 0 else np.finfo(float).eps)
    r_peak = r_seg[i] + d * (r_seg[i+1] - r_seg[i])
    s_peak = y0 - 0.25 * (y_1 - y1) * d
else:
    idx = int(np.argmax(S)); r_peak, s_peak = r[idx], S[idx]

ax.scatter([r_peak], [s_peak], s=40, zorder=5,
           label=rf"max (robust): $S_{{\max}}={s_peak:.3g}$ at $r={r_peak:.3g}$")
ax.axvline(r_peak, ls=":", alpha=0.5); ax.legend(loc="lower right", fontsize=9)
plt.savefig("marshmallow_S_tan.png", dpi=300, bbox_inches="tight"); plt.show()


S_min_diff = np.abs((np.abs(Smin_Sq))**0.5 - S)

fig, ax = plt.subplots()

ax.plot(np.log(r), S_min_diff, label="Difference")
ax.set_xlabel("r")
ax.set_ylabel("Difference")
ax.grid(True)

ax.legend(loc="best")

fig.savefig("marshmallow_difference_S_min.png", dpi=300, bbox_inches="tight")
plt.show()

S_min_diff = np.abs((np.abs(Smin_Sq))**0.5 - S)

fig, ax = plt.subplots()

ax.plot(x, S_min_diff, label="Difference")
ax.set_xlabel("x")
ax.set_ylabel("Difference")
ax.grid(True)

ax.legend(loc="best")

fig.savefig("marshmallow_difference_S_min_xpng", dpi=300, bbox_inches="tight")
plt.show()

S_min_diff = np.abs((np.abs(Smin_Sq))**0.5 - S)

fig, ax = plt.subplots()

ax.plot(r, x, label="X")
ax.set_xlabel("r")
ax.set_ylabel("x(r)")
ax.grid(True)

ax.legend(loc="best")

fig.savefig("marshmallow_x_r.png", dpi=300, bbox_inches="tight")
plt.show()

print(S_min_diff[-1])