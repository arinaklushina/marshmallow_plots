# -----------------------------------------------------------------------------------
# Boson-star BVP solver and plotting script — everything on solve_bvp's mesh
# -----------------------------------------------------------------------------------
# High-level flow:
#   1) Configure Matplotlib + LaTeX-safe saving helpers
#   2) Define tunables, couplings, and mesh controls
#   3) Define the ODE RHS and boundary conditions for solve_bvp
#   4) Build an initial mesh + initial guess and solve the BVP (for base λ)
#   5) Evaluate derived quantities (B, R, derivatives, diagnostics)
#   6) Produce baseline plots
#   7) 2×2 grid: re-solve for four λ values and plot B, B′, 2B″, S, 8P_r, 8S_r^r on EACH panel
#   8) Pairwise grid + profiles and other extras
# -----------------------------------------------------------------------------------

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.collections import LineCollection
from cycler import cycler
from scipy.integrate import solve_bvp
import shutil

# --------------------------- LaTeX / mathtext setup --------------------------------
use_tex = shutil.which("latex") is not None
plt.rcParams.update({"text.usetex": use_tex})
if not use_tex:
    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "mathtext.rm": "serif",
        "mathtext.it": "serif:italic",
        "mathtext.bf": "serif:bold",
    })

def safe_savefig(fig, *args, **kwargs):
    """Save-figure helper robust to constrained_layout + missing LaTeX."""
    if fig.get_constrained_layout() and kwargs.get("bbox_inches") == "tight":
        kwargs = dict(kwargs)
        kwargs.pop("bbox_inches")
    if not shutil.which("latex") and mpl.rcParams.get("text.usetex", False):
        with mpl.rc_context({"text.usetex": False, "mathtext.fontset": "cm"}):
            fig.savefig(*args, **kwargs)
    else:
        fig.savefig(*args, **kwargs)

# --------------------------- Matplotlib global settings ----------------------------
plt.rcParams.update({
    "font.size": 15,
    "font.family": "serif",
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "axes.linewidth": 1.2,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "legend.frameon": False,
    "lines.linewidth": 2.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "axes.prop_cycle": cycler("color", plt.cm.tab10.colors),
})

def prettify(ax):
    ax.minorticks_on()
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))

def smart_ylim(ax, y, q=(0.01, 0.99), pad=0.15):
    yy = np.asarray(y)
    yy = yy[np.isfinite(yy)]
    if yy.size < 5:
        return
    lo, hi = np.quantile(yy, q)
    span = max(hi - lo, 1e-16)
    ax.set_ylim(lo - pad*span, hi + pad*span)

def safe_log_abs(y, floor=1e-300):
    return np.log10(np.maximum(np.abs(y), floor))

# ---------------------------------- tunables --------------------------------------
MK_ON = True
RMAX, NPTS = 1000.0, 8000
EPS_B      = 1e-6

OMEGA = 1.0
SIGMA = 1.0
ALPHA = -1
LAMBDA = 1.0
KAPPA_INF = 5e-2
S0 = 0.4472135954999579

# Hernquist profile bits
M = 10**11       # 10^11 solar masses (dimensionless here)
r0 = 10000.0
rho0 = M / (2*np.pi * r0**3)

if MK_ON:
    ALPHA     = 1/ALPHA/2
    LAMBDA    = LAMBDA/4
    KAPPA_INF = -KAPPA_INF

def rho(r):
    return rho0/(r*(r+1)**3)

def p_r(w, m, gamma, r):
    return 4*ALPHA*(-1 + 6*m*gamma + w**2)/(3*r**4)

def first_param_derivs(r, f):
    rs = np.where(r == 0.0, np.finfo(float).eps, r)
    dw      =  0.5       * rs**3 * f
    dm      = (1.0/12.0) * rs**4 * f
    dgamma  = -0.5       * rs**2 * f
    dkappa  = -(1.0/6.0) * rs    * f
    return dw, dm, dgamma, dkappa

# --------------------------------- RHS / BC (base λ) -------------------------------
def rhs(r, Y):
    """RHS with the (possibly MK-adjusted) global LAMBDA."""
    w, m, gamma, kappa, S, y = Y
    rs = np.where(r == 0.0, np.finfo(float).eps, r)
    S0_asym = np.sqrt(np.clip(-0.5*KAPPA_INF/LAMBDA, 0.0, np.inf))

    B  = w - 2.0*m/rs + gamma*rs - kappa*rs**2
    dB = 2.0*m/rs**2 + gamma - 2.0*kappa*rs
    R  = 2.0*(w - 1.0)/rs**2 + 6.0*gamma/rs - 12.0*kappa

    Babs      = np.maximum(np.abs(B), EPS_B)
    Bsafe     = Babs * np.sign(np.where(B == 0.0, 1.0, B))
    invB      = 1.0 / Bsafe
    dB_over_B = dB / Bsafe

    S_c = np.clip(S, -1e9, 1e9)
    dS  = y / (rs**2 * Bsafe)
    dy  = rs**2 * ((-OMEGA**2 * invB - R/6.0)
                   + (4.0 * LAMBDA / SIGMA + (rho(rs)-p_r(w, m, gamma, rs))/(4*SIGMA*S0_asym**4)) * S_c**2) * S_c

    ddS = dy / (rs**2 * Babs) - dS * (2.0/rs + dB_over_B)

    # Source feeding metric sector (choose model; here CG/Higgs-matter coupling)
    f = -3/(Bsafe*4*ALPHA) * ((S_c/S0_asym)**4 * (rho(rs) + p_r(w, m, gamma, rs)))

    dw, dm, dgamma, dkappa = first_param_derivs(rs, f)
    return np.vstack([dw, dm, dgamma, dkappa, dS, dy])

def bc(Y0, YR):
    """BCs with global LAMBDA."""
    w0, m0, g0, k0, S0_c, y0 = Y0
    wR, mR, gR, kR, SR, yR   = YR
    w0_target = np.sqrt(np.clip(1.0 - 6.0*m0*g0, 0.0, np.inf))
    SR_target = np.sqrt(np.clip(-0.5*KAPPA_INF/LAMBDA, 0.0, np.inf))
    return np.array([
        w0 - w0_target,
        m0 - 0.0,
        g0 - 0.0,
        kR - KAPPA_INF,
        SR - SR_target,
        y0
    ])

# -------------------------------- mesh + initial guess -----------------------------
rmin, rmax = 1e-6, RMAX
r_init = np.geomspace(rmin, rmax, NPTS)

Y_guess = np.zeros((6, r_init.size))
Y_guess[0] = 1.0
Y_guess[2] = 0.0
Y_guess[3] = KAPPA_INF
S_inf = float(np.sqrt(np.clip(-SIGMA * KAPPA_INF/(2.0 * LAMBDA), 0.0, np.inf)))
Y_guess[4] = S_inf
Y_guess[5] = 0.0

# ------------------------------------- solve (base) --------------------------------
sol = solve_bvp(rhs, bc, r_init, Y_guess, tol=1e-10, max_nodes=200000, verbose=2)
if sol.status != 0:
    print("solve_bvp message:", sol.message)

# ---------- EVALUATE EVERYTHING ON THE BVP'S ADAPTIVE MESH ----------
r_bvp = sol.x
w, m, gamma, kappa, S, y = sol.y
rs = np.where(r_bvp == 0.0, np.finfo(float).eps, r_bvp)

B   = w - 2.0*m/rs + gamma*rs - kappa*rs**2
dB  = 2.0*m/rs**2 + gamma - 2.0*kappa*rs
R   = 2.0*(w - 1.0)/rs**2 + 6.0*gamma/rs - 12.0*kappa

Babs      = np.maximum(np.abs(B), EPS_B)
Bsafe     = Babs * np.sign(np.where(B == 0.0, 1.0, B))
invB      = 1.0 / Bsafe
dB_over_B = dB / Bsafe

S_c = np.clip(S, -1e9, 1e9)
dS  = y / (rs**2 * Babs)
dy  = -rs**2 * ((OMEGA**2 * invB - R/6.0) - (4.0*LAMBDA / SIGMA)*S_c**2) * S_c
ddS = dy / (rs**2 * Babs) - dS * (2.0/rs + dB_over_B)

# Diagnostics (keep consistent with your chosen rhs model if needed)
f = (-SIGMA / (4.0 * ALPHA)) * (-2.0 * (dS**2) + ddS * S_c - OMEGA**2 * S_c**2 * invB**2)
fp  = np.gradient(f,  r_bvp, edge_order=2)
fpp = np.gradient(fp, r_bvp, edge_order=2)

Bp   =  2.0*m/rs**2 + gamma - 2.0*kappa*rs
Bpp  = -4.0*m/rs**3 - 2.0*kappa
Bppp = 12.0*m/rs**4

Smin_Sq  = - 2.0 * (-R/12.0 + OMEGA**2) / (4.0 * LAMBDA / SIGMA)

# ------------------------------ plotting: B and derivs -----------------------------
fig, ax1 = plt.subplots(constrained_layout=True)
l1 = ax1.plot(r_bvp, B, label=r"$B(r)$")[0]
ax1.set_xlabel(r"$r$")
ax1.set_ylabel(r"$B(r)$", color=l1.get_color())
ax1.tick_params(axis="y", colors=l1.get_color())
ax1.axhline(0, lw=1, alpha=0.4)
prettify(ax1); smart_ylim(ax1, B)

ax2 = ax1.twinx()
l2 = ax2.plot(r_bvp, Bp,  label=r"$B'(r)$", color='purple')[0]
ax2.plot(r_bvp, Bpp, label=r"$B''(r)$", linestyle="--", color='green')
ax2.plot(r_bvp, Bppp,label=r"$B'''(r)$", linestyle=":",  color='yellow')
ax2.set_ylabel(r"$B',\,B'',\,B'''$", color=l2.get_color())
ax2.tick_params(axis="y", colors=l2.get_color())
prettify(ax2); smart_ylim(ax2, np.c_[Bp, Bpp, Bppp].ravel())

L1,N1 = ax1.get_legend_handles_labels()
L2,N2 = ax2.get_legend_handles_labels()
ax1.legend(L1+L2, N1+N2, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)
safe_savefig(fig, 'marshmallow_B.pdf', bbox_inches='tight', metadata={"Title":"B and derivatives"})
safe_savefig(fig, 'marshmallow_B.png', dpi=300, bbox_inches='tight')

# ----------------------------------- plot: f(r) ------------------------------------
fig, ax = plt.subplots(constrained_layout=True)
ax.axhline(0, lw=1, alpha=0.4)
ax.set_xlim(0, 10)
ax.set_ylim(-10, 10)
ax.plot(r_bvp, f, label=r"$f(r)$")
ax.set_xlabel(r"$r$"); ax.set_ylabel(r"$f(r)$")
prettify(ax); smart_ylim(ax, f)
safe_savefig(fig, 'marshmallow_f.png', dpi=300, bbox_inches='tight')

# -------------------------- potential & curvature diagnostics ----------------------
fig, ax = plt.subplots(constrained_layout=True)
ax.axhline(0, lw=1, alpha=0.4)
ax.plot(r_bvp, R, label=r"$R(r)$")
ax.set_xlabel(r"$r$"); ax.set_ylabel(r"$R(r)$")
prettify(ax); smart_ylim(ax, R)
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
safe_savefig(fig, "marshmallow_R.pdf", bbox_inches="tight", metadata={"Title":"R(r)"})
safe_savefig(fig, "marshmallow_R.png", dpi=300, bbox_inches="tight")

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(r_bvp, Smin_Sq, "--", label=r"$S_{\min}^2(r)$")
ax.plot(r_bvp, S**2,   "-",  label=r"$S^2(r)$")
ax.set_xlabel(r"$r$"); ax.set_ylabel(r"$S_{\min}^2,\,S^2$")
prettify(ax); smart_ylim(ax, np.c_[Smin_Sq, S**2].ravel())
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
safe_savefig(fig, "marshmallow_S_min_sq_and_S_sq.pdf", bbox_inches="tight",
             metadata={"Title":"S^2 vs Smin^2"})
safe_savefig(fig, "marshmallow_S_min_sq_and_S_sq.png", dpi=300, bbox_inches="tight")

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(np.log10(r_bvp), safe_log_abs(f), label=r"$\log_{10}|f|$")
ax.set_xlabel(r"$\log_{10} r$"); ax.set_ylabel(r"$\log_{10}|f|$")
prettify(ax)
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
safe_savefig(fig, "marshmallow_f_log10.pdf", bbox_inches="tight", metadata={"Title":"log10|f|"})
safe_savefig(fig, "marshmallow_f_log10.png", dpi=300, bbox_inches="tight")

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(r_bvp, S, linewidth=2.0, label=r"$S(r)$")
ax.set_xlabel(r"$r$"); ax.set_ylabel(r"$S(r)$")
ax.axhline(0, lw=1, alpha=0.4)
prettify(ax)
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
safe_savefig(fig, "marshmallow_S_solid.pdf", bbox_inches="tight", metadata={"Title":"S(r) solid"})
safe_savefig(fig, "marshmallow_S_solid.png", dpi=300, bbox_inches="tight")

plt.show()

# ---------------------- solver factory for arbitrary λ -----------------------------
def rhs_factory(lam):
    """Closure that builds the RHS with a specific lambda 'lam'."""
    def _rhs(r, Y):
        w, m, gamma, kappa, S, y = Y
        rs = np.where(r == 0.0, np.finfo(float).eps, r)
        S0_asym = np.sqrt(np.clip(-0.5*KAPPA_INF/lam, 0.0, np.inf))

        B  = w - 2.0*m/rs + gamma*rs - kappa*rs**2
        dB = 2.0*m/rs**2 + gamma - 2.0*kappa*rs
        R  = 2.0*(w - 1.0)/rs**2 + 6.0*gamma/rs - 12.0*kappa

        Babs      = np.maximum(np.abs(B), EPS_B)
        Bsafe     = Babs * np.sign(np.where(B == 0.0, 1.0, B))
        invB      = 1.0 / Bsafe
        dB_over_B = dB / Bsafe

        S_c = np.clip(S, -1e9, 1e9)
        dS  = y / (rs**2 * Bsafe)
        dy  = rs**2 * ((-OMEGA**2 * invB - R/6.0)
                       + (4.0 * lam / SIGMA + (rho(rs)-p_r(w, m, gamma, rs))/(4*SIGMA*S0_asym**4)) * S_c**2) * S_c

        # source (same model as base)
        f = -3/(Bsafe*4*ALPHA) * ((S_c/S0_asym)**4 * (rho(rs) + p_r(w, m, gamma, rs)))
        dw, dm, dgamma, dkappa = first_param_derivs(rs, f)
        return np.vstack([dw, dm, dgamma, dkappa, dS, dy])
    return _rhs

def bc_factory(lam):
    def _bc(Y0, YR):
        w0, m0, g0, k0, S0_c, y0 = Y0
        wR, mR, gR, kR, SR, yR   = YR
        w0_target = np.sqrt(np.clip(1.0 - 6.0*m0*g0, 0.0, np.inf))
        SR_target = np.sqrt(np.clip(-0.5*KAPPA_INF/lam, 0.0, np.inf))
        return np.array([w0 - w0_target, m0, g0, kR - KAPPA_INF, SR - SR_target, y0])
    return _bc

def solve_for_lambda(lam):
    """
    Solve the BVP for a given lambda=lam and return all arrays needed for plotting.
    Returns:
      r, B, Bp, Bpp, S, dS, dB, m, gamma, w
    """
    # initial guess tuned to this lam
    Y0 = np.zeros_like(Y_guess)
    Y0[0] = 1.0
    Y0[2] = 0.0
    Y0[3] = KAPPA_INF
    S_inf_lam = float(np.sqrt(np.clip(-SIGMA * KAPPA_INF/(2.0 * lam), 0.0, np.inf)))
    Y0[4] = S_inf_lam
    Y0[5] = 0.0

    sol_lam = solve_bvp(rhs_factory(lam), bc_factory(lam), r_init, Y0,
                        tol=1e-10, max_nodes=200000, verbose=0)
    if sol_lam.status != 0:
        print(f"[λ={lam:g}] solve_bvp:", sol_lam.message)

    r = sol_lam.x
    w_i, m_i, g_i, k_i, S_i, y_i = sol_lam.y
    rs_i = np.where(r == 0.0, np.finfo(float).eps, r)

    B_i  = w_i - 2.0*m_i/rs_i + g_i*rs_i - k_i*rs_i**2
    dB_i = 2.0*m_i/rs_i**2 + g_i - 2.0*k_i*rs_i
    Bp_i = dB_i
    Bpp_i = -4.0*m_i/rs_i**3 - 2.0*k_i

    Babs_i = np.maximum(np.abs(B_i), EPS_B)
    dS_i   = y_i / (rs_i**2 * Babs_i)

    return r, B_i, Bp_i, Bpp_i, S_i, dS_i, dB_i, m_i, g_i, w_i

# ---------------------- 2x2 grid: FULL re-solve per λ, same curves each panel ------
fig, axs = plt.subplots(2, 4, figsize=(10, 8), sharex=False, sharey=False, constrained_layout=True)
axes = axs.ravel()

LAMBDA0 = float(LAMBDA)
lam_vals = [LAMBDA0*10, LAMBDA0, LAMBDA0*0.5, LAMBDA0*0.3, LAMBDA0*0.25, LAMBDA0*0.243, LAMBDA0*0.241, LAMBDA0*0.24]

handles = None
legend_labels  = [r"$B$", r"$B'$", r"$2B''$", r"$S$", r"$8P_r$", r"$8S_r^r$"]

for ax, lam in zip(axes, lam_vals):
    r_i, B_i, Bp_i, Bpp_i, S_i, dS_i, dB_i, m_i, g_i, w_i = solve_for_lambda(lam)
    rs_i = np.where(r_i == 0.0, np.finfo(float).eps, r_i)

    p_r_i = 4*ALPHA*(-1 + 6*m_i*g_i + w_i**2) / (3*rs_i**4)
    S_rr_i = (OMEGA**2 * S_i**2 / (6*B_i) + B_i/2*dS_i**2
              + S_i*dS_i*(dB_i + 4*B_i/rs_i)/6
              + S_i**2*(dB_i/rs_i + (B_i - 1)/rs_i**2)/6
              - lam*S_i**4 / SIGMA)
    S_rr_i = SIGMA * S_rr_i

    hs = [
        ax.plot(r_i, B_i,                        label=r"$B$")[0],
        ax.plot(r_i, Bp_i,                       label=r"$B'$", linestyle="--")[0],
        ax.plot(r_i, 2*Bpp_i,                    label=r"$2B''$", linestyle=":")[0],
        ax.plot(r_i, S_i/(OMEGA/(-ALPHA)**0.5),  label=r"$S$")[0],
        ax.plot(r_i, 8*p_r_i,                    label=r"$8P_r$", color='yellow')[0],
        ax.plot(r_i, 8*S_rr_i,                   label=r"$8S_r^r$", color='magenta')[0],
    ]
    ax.axhline(0, lw=1, alpha=0.35)
    ax.set_xlim(r_i.min(), min(10.0, r_i.max()))
    ax.set_ylim(-1.5, 3)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$B,\,B',\,2B'',\,S,\,8P_r,\,8S_r^{\,r}$")
    ax.set_title(fr"$\lambda = {lam:g}$")
    prettify(ax)

    if handles is None: handles = hs

fig.legend(handles, legend_labels, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)
safe_savefig(fig, "grid_v_lambda_full_resolve.pdf", bbox_inches="tight")
safe_savefig(fig, "grid_v_lambda_full_resolve.png", dpi=300, bbox_inches="tight")
plt.show()

# ================================ GRID PLOT BLOCK ==================================
# Pairwise grid + bottom profiles, all on r_bvp (base λ)
names  = ["w", "m", "gamma", "kappa", "S", "y"]
arrays = [w,    m,   gamma,   kappa,   S,   y]

origin_vals = []
for i in range(4):
    print(names[i], "=", arrays[i][0], "at the origin-ish mesh point")
    origin_vals.append(arrays[i][0])

R_origin = 2*(origin_vals[0] - 1) / r_bvp[0]**2 + 6 * origin_vals[2] / r_bvp[0] - 12 * origin_vals[3]
print("R at the origin-ish mesh point is", R_origin)

fig = plt.figure(figsize=(18, 21))
gs  = fig.add_gridspec(nrows=7, ncols=6, height_ratios=[1]*6 + [1.1])

axes_grid = [[fig.add_subplot(gs[i, j]) for j in range(6)] for i in range(6)]
for i in range(6):
    for j in range(6):
        ax = axes_grid[i][j]
        if i == j:
            ax.text(0.5, 0.5, names[i], ha="center", va="center", fontsize=12, weight="bold")
            ax.set_xticks([]); ax.set_yticks([])
        elif i < j:
            ax.set_xticks([]); ax.set_yticks([])
        else:
            xi = arrays[j]; yi = arrays[i]
            valid = np.isfinite(xi) & np.isfinite(yi)
            sc = ax.scatter(xi[valid], yi[valid], c=r_bvp[valid], s=5, cmap="viridis")
            if i == 5:  ax.set_xlabel(names[j])
            if j == 0:  ax.set_ylabel(names[i])
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

labels_bottom = [r"$w(r)$", r"$m(r)$", r"$\gamma(r)$", r"$\kappa(r)$", r"$S(r)$", r"$y(r)$"]
for j, (lab, arr) in enumerate(zip(labels_bottom, arrays)):
    ax = fig.add_subplot(gs[6, j])
    ax.plot(r_bvp, arr)
    ax.set_title(lab)
    ax.set_xlabel(r"$r$")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

plt.suptitle("All variable pairs (color = r) — evaluated on solve_bvp mesh", fontsize=16)
plt.tight_layout(rect=[0, 0, 0.95, 0.95])
safe_savefig(fig, "marshmallow_all_pairs_with_profiles.png", dpi=300, bbox_inches="tight")
plt.show()

print("S_0 (first mesh point) = ", S[0])
print("final value of S", S[-1])

# ====== VELOCITY PROFILE =======
M = 10**11 * 1.989e33
gamma_cm = 8.5243e-30
k_cm = 9.54e-54
c = 2.99792458e10
G = 6.67430e-8
kpc_to_cm = 3.0857e21
kmps_to_kpcps = 1 / (c / 1e5)

def v_B_sq(B, Bp, r): return r*Bp/(2*B)
def v_S_sq(S, Sp, r): return r*Sp/S
def v_sq(v_B_sq, v_S_sq): return (v_B_sq + v_S_sq) / (1 + v_S_sq)

v_B_sq_ = r_bvp * Bp / (2 * B)
v_S_sq_ = r_bvp * dS / S
v_sq_   = (v_B_sq_ + v_S_sq_) / (1 + v_S_sq_)
v_mag   = np.sqrt(np.maximum(v_sq_, 1e-30))

x_values = [0.2, 0.38, 0.66, 1.61, 2.57, 3.59, 4.51, 5.53, 6.50, 7.56, 8.34, 9.45,
            10.50, 11.44, 12.51, 13.53, 14.59, 16.05, 18.64, 26.30, 28.26,
            29.51, 32.04, 33.99, 36.49, 38.41, 40.42, 42.40, 44.49, 45.99, 48.06,
            49.49, 51.39, 53.89, 56.89, 57.98, 60.92, 64.73, 69.31, 72.96, 76.95,
            81.13, 84.90, 89.35, 92.44, 97.41, 100.72, 106.77, 119.98, 189.49]
y_values_kmps = [233.0, 268.92, 250.75, 217.83, 219.58, 223.11, 247.88, 253.14, 270.95, 267.80,
                 270.52, 235.58, 249.72, 261.96, 284.30, 271.54, 251.43, 320.70, 286.46, 189.64,
                 237.99, 209.82, 179.14, 170.37, 175.92, 191.57, 197.59, 192.79, 213.22, 179.39,
                 213.03, 178.57, 183.31, 157.89, 191.76, 210.72, 168.02, 206.47, 203.62, 190.53,
                 222.72, 186.29, 122.25, 143.95, 154.66, 184.0, 108.68, 137.15, 150.18, 125.01]
y_values = [y * kmps_to_kpcps for y in y_values_kmps]

mask_model = (r_bvp > 0) & (v_mag > 0)
rm = r_bvp[mask_model]
vm = v_mag[mask_model]

import numpy as np
mask_data = (np.array(x_values) > 0) & (np.array(y_values) > 0)
xd = np.array(x_values)[mask_data]
yd = np.array(y_values)[mask_data]

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(np.log10(rm), np.log10(vm), lw=2.0, label="model")
ax.scatter(np.log10(xd), np.log10(yd), s=28, facecolors='none', edgecolors='crimson', label="MW")
ax.set_xlabel(r"$\log_{10} r$")
ax.set_ylabel(r"$\log_{10} v$")
ax.legend()
safe_savefig(fig, "marshmallow_v_loglog_dumb.pdf",  bbox_inches="tight")
safe_savefig(fig, "marshmallow_v_loglog_dumb.png", dpi=300, bbox_inches="tight")
plt.show()

# ------------------------------ compare B(r) to GR ------------------------------
M_solar = 1.989e33
M_phys  = 1.0e11 * M_solar
G_cgs   = 6.67430e-8
c_cgs   = 2.99792458e10
kpc_to_cm = 3.0857e21

r_cm = np.maximum(r_bvp, np.finfo(float).eps) * kpc_to_cm
B_GR = 1.0 - 2.0 * G_cgs * M_phys / (r_cm * c_cgs**2)

fig, ax = plt.subplots(constrained_layout=True)
ax.axhline(0, lw=1, alpha=0.35)
ax.plot(r_bvp, B, lw=2.0, label=r"$B(r)$ (model)")
ax.plot(r_bvp, B_GR, lw=2.0, linestyle="--", label=r"$B_{\mathrm{GR}}(r)=1-\frac{2GM}{rc^2}$")
ax.set_xlabel(r"$r\ \mathrm{[kpc]}$")
ax.set_ylabel(r"$B(r)$")
prettify(ax)
ax.set_xlim(0,10)
ax.set_ylim(0, 10)
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
safe_savefig(fig, "marshmallow_B_vs_GR.png", dpi=300, bbox_inches="tight")
plt.show()
