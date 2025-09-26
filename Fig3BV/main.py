# -----------------------------------------------------------------------------------
# Boson-star BVP solver and plotting script — everything on solve_bvp's mesh
# -----------------------------------------------------------------------------------
# High-level flow:
#   1) Configure Matplotlib + LaTeX-safe saving helpers
#   2) Define tunables, couplings, and mesh controls
#   3) Define the ODE RHS and boundary conditions for solve_bvp
#   4) Build an initial mesh + initial guess and solve the BVP
#   5) Evaluate derived quantities (B, R, derivatives, f diagnostics)
#   6) Produce plots (B and derivatives, f(r), R(r), S_min^2 vs S^2, etc.)
#   7) Extra combined plot with B, B', 2B'', S, 8P_r (as requested)
#   8) (Optional) Pairwise grid of variables and profile plots

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.collections import LineCollection
from cycler import cycler
from scipy.integrate import solve_bvp
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import shutil
import matplotlib as mpl

# --------------------------- LaTeX / mathtext setup --------------------------------
# Detect availability of a LaTeX installation and set text rendering accordingly.
use_tex = shutil.which("latex") is not None
plt.rcParams.update({"text.usetex": use_tex})
if not use_tex:
    # Keep the Computer Modern look via mathtext if LaTeX is not available.
    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "mathtext.rm": "serif",
        "mathtext.it": "serif:italic",
        "mathtext.bf": "serif:bold",
    })

def safe_savefig(fig, *args, **kwargs):
    """
    Save figure; if TeX isn't available but usetex=True in rcParams,
    temporarily disable usetex so saving never crashes.

    Also avoid bbox_inches='tight' when constrained_layout is active,
    which otherwise can collapse axes to zero size.
    """
    # If constrained_layout is active, strip bbox_inches='tight'
    if fig.get_constrained_layout() and kwargs.get("bbox_inches") == "tight":
        kwargs = dict(kwargs)  # shallow copy to avoid mutating caller's dict
        kwargs.pop("bbox_inches")

    # TeX-safe saving (fallback to mathtext if LaTeX unavailable)
    if not shutil.which("latex") and mpl.rcParams.get("text.usetex", False):
        with mpl.rc_context({"text.usetex": False, "mathtext.fontset": "cm"}):
            fig.savefig(*args, **kwargs)
    else:
        fig.savefig(*args, **kwargs)

# --------------------------- Matplotlib global settings ----------------------------
# Plot aesthetics (tick directions, grid, linewidths, etc.).
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
    "axes.prop_cycle": cycler("color", plt.cm.tab10.colors),  # colorblind-friendly
})

def prettify(ax):
    """Turn on minor ticks and use math-aware scalar format on axes."""
    ax.minorticks_on()
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))

def smart_ylim(ax, y, q=(0.01, 0.99), pad=0.15):
    """
    Set y-limits based on central quantiles (q) with padding to avoid clipping.
    Skips if too few finite points are present.
    """
    yy = np.asarray(y)
    yy = yy[np.isfinite(yy)]
    if yy.size < 5:
        return
    lo, hi = np.quantile(yy, q)
    span = max(hi - lo, 1e-16)
    ax.set_ylim(lo - pad*span, hi + pad*span)

def safe_log_abs(y, floor=1e-300):
    """Compute log10|y| with a numerical floor to avoid -inf."""
    return np.log10(np.maximum(np.abs(y), floor))

def colored_line(ax, x, y, c, cmap="viridis", lw=2.0):
    """Draw a line whose color varies with c (same length as x/y)."""
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, array=np.asarray(c)[:-1], cmap=cmap)
    lc.set_linewidth(lw)
    h = ax.add_collection(lc)
    ax.set_xlim(np.min(x), np.max(x))
    smart_ylim(ax, y)
    return h  # handle for colorbar

# def extrema_from_fp(x, y, fp):
#     """
#     Find extrema of y(x) from zero-crossings of fp and refine with a quadratic fit
#     on a 3-point window.
#     """
#     x = np.asarray(x); y = np.asarray(y); fp = np.asarray(fp)
#     zc = np.where(np.diff(np.sign(fp)) != 0)[0]
#     xs, ys = [], []
#     for i in zc:
#         j = int(np.clip(i, 1, len(x) - 2))
#         xw, yw = x[j-1:j+1+1], y[j-1:j+1+1]
#         A = np.vstack([xw**2, xw, np.ones_like(xw)]).T
#         a, b, c = np.linalg.lstsq(A, yw, rcond=None)[0]
#         if abs(a) > 0:
#             x_star = -b / (2*a)
#             x_star = float(np.clip(x_star, xw.min(), xw.max()))
#             y_star = float(a*x_star**2 + b*x_star + c)
#         else:
#             x_star, y_star = float(x[j]), float(y[j])
#         xs.append(x_star); ys.append(y_star)
#     return np.array(xs), np.array(ys)

# ---------------------------------- tunables --------------------------------------

MK_ON = True
RMAX, NPTS = 100.0, 8000
EPS_B      = 1e-6

OMEGA = 1.0
SIGMA = 1.0
ALPHA = -1
LAMBDA = 1
KAPPA_INF = 0.05

if MK_ON:
    ALPHA   = 1/ALPHA/2
    LAMBDA  = LAMBDA/4
    KAPPA_INF = -KAPPA_INF


# ----------------------- helpers for parameter derivatives -------------------------
def first_param_derivs(r, f):
    """
    Given source f(r), compute the derivatives for (w, m, gamma, kappa)
    used in the parametrized form of the metric function B.
    """
    rs = np.where(r == 0.0, np.finfo(float).eps, r)
    dw      =  0.5       * rs**3 * f
    dm      = (1.0/12.0) * rs**4 * f
    dgamma  = -0.5       * rs**2 * f
    dkappa  = -(1.0/6.0) * rs    * f
    return dw, dm, dgamma, dkappa

# ------------------------------ BVP RHS -------------------------------------------
# State Y = [w, m, gamma, kappa, S, y]
def rhs(r, Y):
    """
    Right-hand side of the ODE system for solve_bvp.
    The metric ansatz is encoded via B(w,m,gamma,kappa; r).
    The scalar sector uses S and y where y = r^2 B S'.
    """
    w, m, gamma, kappa, S, y = Y
    rs = np.where(r == 0.0, np.finfo(float).eps, r)

    # metric combinations
    B  = w - 2.0*m/rs + gamma*rs - kappa*rs**2
    dB = 2.0*m/rs**2 + gamma - 2.0*kappa*rs
    R  = 2.0*(w - 1.0)/rs**2 + 6.0*gamma/rs - 12.0*kappa

    # numerically-safe helpers (sign-preserving clamp)
    Babs      = np.maximum(np.abs(B), EPS_B)
    Bsafe     = Babs * np.sign(np.where(B == 0.0, 1.0, B))
    invB      = 1.0 / Bsafe
    dB_over_B = dB / Bsafe

    # scalar field eqns
    S_c = np.clip(S, -1e9, 1e9)
    dS  = y / (rs**2 * Bsafe)  # equivalent to S'
    dy  = rs**2* ((-OMEGA**2 * invB - R/6.0) + (4.0 * LAMBDA / SIGMA) * S_c**2) * S_c

    # algebraic ddS for source construction
    ddS = dy / (rs**2 * Babs) - dS * (2.0/rs + dB_over_B)

    # source from S and its derivatives (feeds into (w,m,gamma,kappa) evolution)
    f = (-SIGMA / (4.0 * ALPHA)) * (-2.0 * (dS**2)
                                    + ddS * S_c - OMEGA**2 * S_c**2 * invB**2)
    # feed back into metric derivatives
    dw, dm, dgamma, dkappa = first_param_derivs(rs, f)

    return np.vstack([dw, dm, dgamma, dkappa, dS, dy])

# ------------------------------ Boundary conditions --------------------------------
def bc(Y0, YR):
    """
    Boundary conditions at r=0 and r=RMAX:
      - w(0) constrained via regularity relation with m(0), gamma(0)
      - m(0)=0, gamma(0)=0 (regular center)
      - kappa(RMAX)=KAPPA_INF (asymptotics)
      - S(RMAX) ~ sqrt(-0.5 * KAPPA_INF / LAMBDA)
      - y(0)=0 (i.e., S'(0)=0)
    """
    # Y = [w, m, gamma, kappa, S, y]
    w0, m0, g0, k0, S0, y0 = Y0          # r=0
    wR, mR, gR, kR, SR, yR = YR          # r=RMAX ~ ∞
    # Clip sqrt args to avoid ComplexWarning from tiny negatives:
    w0_target = np.sqrt(np.clip(1.0 - 6.0*m0*g0 + 3*rmin**4* 0.03230628138243277/4 / ALPHA , 0.0, np.inf))
    SR_target = np.sqrt(np.clip(-0.5*KAPPA_INF/LAMBDA, 0.0, np.inf))
    return np.array([
        w0 - w0_target,
        m0 - 0.0,
        g0 - 0.0,
        kR - KAPPA_INF,   # far field curvature
        SR - SR_target,   # asymptotic S expect: 0.447
        y0                # S'(0)=0
    ])


# -------------------------------- mesh + initial guess -----------------------------
# Geometric grid from 10^-5 to 10^3 (uniform in log10 r)
RMAX, NPTS = 1e3, NPTS          # keep NPTS as you set above; just ensure RMAX = 1e3
rmin, rmax = 1e-6, RMAX
r_init = np.geomspace(rmin, rmax, NPTS)

# Gentle initial guess
Y_guess = np.zeros((6, r_init.size))
Y_guess[0] = 1.0               # w ~ 1
Y_guess[2] = 0.0               # gamma ~ 0
Y_guess[3] = KAPPA_INF         # kappa ~ kappa_inf
S_inf = (np.abs(-SIGMA * KAPPA_INF/ (2 * LAMBDA)))**0.5
Y_guess[4] = S_inf             # small central S (using far-field seed)
Y_guess[5] = 0.0               # y seed

# ------------------------------------- solve ---------------------------------------
# High-accuracy tolerance and large max_nodes (can be slow, but robust).
sol = solve_bvp(rhs, bc, r_init, Y_guess, tol=1e-10, max_nodes=200000, verbose=2)
if sol.status != 0:
    print("solve_bvp message:", sol.message)

# ---------- EVALUATE EVERYTHING ON THE BVP'S ADAPTIVE MESH ----------
# sol.x : (n_points,) adaptive mesh; sol.y : (6, n_points) solution
r_bvp = sol.x
w, m, gamma, kappa, S, y = sol.y

# --- enforce algebraic constraint WITHOUT touching the ODEs/BCs ---
# Constraint: 1 - w**2 = 6*m*gamma
# Strategy: where |gamma| is not tiny, project m := (1 - w**2)/(6*gamma).
# # Where |gamma| is tiny, project w to satisfy the constraint with existing m (take sign of current w).
# EPS_CONS = 1e-12
# mask = np.abs(gamma) > EPS_CONS
# m = np.where(mask, (1.0 - w**2) / (6.0 * gamma), m)
# w = np.where(~mask, np.sign(w) * np.sqrt(np.maximum(1.0 - 6.0*m*gamma, 0.0)), w)
# cons_res = 1.0 - w**2 - 6.0*m*gamma
# print("Constraint max |residual| after projection:", np.nanmax(np.abs(cons_res)))

# Numerically safe radius array (avoid division by zero at r=0).
rs = np.where(r_bvp == 0.0, np.finfo(float).eps, r_bvp)

# Metric combos on the BVP mesh
B   = w - 2.0*m/rs + gamma*rs - kappa*rs**2
dB  = 2.0*m/rs**2 + gamma - 2.0*kappa*rs
R   = 2.0*(w - 1.0)/rs**2 + 6.0*gamma/rs - 12.0*kappa

# Safe helpers
Babs      = np.maximum(np.abs(B), EPS_B)
Bsafe     = Babs * np.sign(np.where(B == 0.0, 1.0, B))
invB      = 1.0 / Bsafe
dB_over_B = dB / Bsafe

# Scalar pieces on the BVP mesh (post-solve diagnostics)
S_c = np.clip(S, -1e9, 1e9)
dS  = y / (rs**2 * Babs)
dy  = -rs**2 * ((OMEGA**2 * invB - R/6.0) - (4.0*LAMBDA / SIGMA)*S_c**2) * S_c
ddS = dy / (rs**2 * Babs) - dS * (2.0/rs + dB_over_B)

# Source and numerical derivatives on the BVP mesh
# (stay consistent with rhs for f)
f = (-SIGMA / (4.0 * ALPHA)) * (-2.0 * (dS**2) + ddS * S_c - OMEGA**2 * S_c**2 * invB**2)
# IMPORTANT: gradient w.r.t. NON-UNIFORM spacing r_bvp
fp  = np.gradient(f,  r_bvp, edge_order=2)
fpp = np.gradient(fp, r_bvp, edge_order=2)

# Closed-form B-derivatives (on BVP mesh)
Bp   =  2.0*m/rs**2 + gamma - 2.0*kappa*rs
Bpp  = -4.0*m/rs**3 - 2.0*kappa
Bppp = 12.0*m/rs**4

# Potential diagnostics on the same mesh
# V        = (-R/12.0 + OMEGA**2) * S**2 + (LAMBDA / SIGMA) * S**4
Smin_Sq  = - 2.0 * (-R/12.0 + OMEGA**2) / (4.0 * LAMBDA / SIGMA)

# ------------------------------ plotting: B, B', B'', B''' -------------------------
# Figure with B and its first three r-derivatives (B', B'', B''') on twin y-axes.
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
# Source profile f(r) with fixed x/y windows plus smart y-limits overlay.
fig, ax = plt.subplots(constrained_layout=True)
ax.axhline(0, lw=1, alpha=0.4)
ax.set_xlim(0, 10)
ax.set_ylim(-10, 10)
lf_line = ax.plot(r_bvp, f, label=r"$f(r)$")[0]
ax.set_xlabel(r"$r$"); ax.set_ylabel(r"$f(r)$")
prettify(ax)
smart_ylim(ax, f)

safe_savefig(fig, 'marshmallow_f.png', dpi=300, bbox_inches='tight')

# -------------------------- potential & curvature diagnostics ----------------------
# R(r)
fig, ax = plt.subplots(constrained_layout=True)
ax.axhline(0, lw=1, alpha=0.4)
ax.plot(r_bvp, R, label=r"$R(r)$")
ax.set_xlabel(r"$r$"); ax.set_ylabel(r"$R(r)$")
prettify(ax); smart_ylim(ax, R)
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
safe_savefig(fig, "marshmallow_R.pdf", bbox_inches="tight", metadata={"Title":"R(r)"})
safe_savefig(fig, "marshmallow_R.png", dpi=300, bbox_inches="tight")

# S_min^2 and S^2
# Compare local minimum of potential (S_min^2) to the actual S^2(r) profile.
fig, ax = plt.subplots(constrained_layout=True)
ax.plot(r_bvp, Smin_Sq, "--", label=r"$S_{\min}^2(r)$")
ax.plot(r_bvp, S**2,   "-",  label=r"$S^2(r)$")
ax.set_xlabel(r"$r$"); ax.set_ylabel(r"$S_{\min}^2,\,S^2$")
prettify(ax); smart_ylim(ax, np.c_[Smin_Sq, S**2].ravel())
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
safe_savefig(fig, "marshmallow_S_min_sq_and_S_sq.pdf", bbox_inches="tight",
             metadata={"Title":"S^2 vs Smin^2"})
safe_savefig(fig, "marshmallow_S_min_sq_and_S_sq.png", dpi=300, bbox_inches="tight")

# # V(S) scatter colored by r (color uses r_bvp)
# valid = np.isfinite(S) & np.isfinite(V) & np.isfinite(r_bvp)
# fig, ax = plt.subplots(constrained_layout=True)
# sc = ax.scatter(S[valid], V[valid], c=r_bvp[valid], s=12, alpha=0.85, cmap="viridis")
# ax.set_xlabel(r"$S$"); ax.set_ylabel(r"$V(S)$")
# prettify(ax); fig.colorbar(sc, ax=ax, label=r"$r$")
# safe_savefig(fig, "marshmallow_Mexican_hat.pdf", bbox_inches="tight", metadata={"Title":"V(S)"})
# safe_savefig(fig, "marshmallow_Mexican_hat.png", dpi=300, bbox_inches="tight")

# f on log axes (optional alternative view)
fig, ax = plt.subplots(constrained_layout=True)
ax.plot(np.log10(r_bvp), safe_log_abs(f), label=r"$\log_{10}|f|$")
ax.set_xlabel(r"$\log_{10} r$"); ax.set_ylabel(r"$\log_{10}|f|$")
prettify(ax)
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
safe_savefig(fig, "marshmallow_f_log10.pdf", bbox_inches="tight", metadata={"Title":"log10|f|"})
safe_savefig(fig, "marshmallow_f_log10.png", dpi=300, bbox_inches="tight")

# --- a single-panel curve for S(r)  ---
fig, ax = plt.subplots(constrained_layout=True)
ax.plot(r_bvp, S, linewidth=2.0, label=r"$S(r)$")
ax.set_xlabel(r"$r$"); ax.set_ylabel(r"$S(r)$")
ax.axhline(0, lw=1, alpha=0.4)
prettify(ax)
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
safe_savefig(fig, "marshmallow_S_solid.pdf", bbox_inches="tight", metadata={"Title":"S(r) solid"})
safe_savefig(fig, "marshmallow_S_solid.png", dpi=300, bbox_inches="tight")

plt.show()

# ---------------------------- boundary value utility (optional) --------------------
def value_at_RMAX(Rarr, rarr, RMAX, mode="interp"):
    """
    Convenience helper to extract a value at RMAX from an array R(r):
      - edge: last sample
      - max:  global maximum
      - interp: linear interpolation at exactly RMAX
    """
    if mode == "edge":
        return Rarr[-1]
    if mode == "max":
        return float(np.max(Rarr))
    if mode == "interp":
        return float(np.interp(RMAX, rarr, Rarr))
    raise ValueError("unknown mode")

# Example asymptotic scalar estimate using interpolated R(RMAX).
R_max = value_at_RMAX(R, r_bvp, RMAX, mode="interp")
val = (SIGMA / (2 * LAMBDA)) * np.sqrt(R_max/12.0 + OMEGA**2)
print("Asymptotic S target value:", val)


# ---------------------- combined plot: B, B', B'', S, Pr, Srr ---------------------------
fig, axL = plt.subplots(constrained_layout=True)

# Local pressure ingredients (Higgs contribution S_rr and total radial pressure p_r).
S_rr = (OMEGA**2 * S**2 / (6*B) + B/2*dS**2 + S*dS*(dB + 4*B/rs)/6
        + S**2*(dB/rs + (B - 1)/rs**2)/6 - LAMBDA*S**4 / SIGMA) # Higgs contribution to pressure
S_rr = SIGMA*S_rr
p_r =  4*ALPHA*(-1 + 6*m*gamma + w**2)/(3*rs**4)


# Left axis: B, B', B'', S, pr
hB   = axL.plot(r_bvp, B,   label=r"$B$",   linewidth=2.0)[0]
hBp  = axL.plot(r_bvp, Bp,  label=r"$B'$",  linestyle="--")[0]
hBpp = axL.plot(r_bvp, 2*Bpp, label=r"$2B''$", linestyle=":")[0]  # scaled by 2 for visibility
hS = axL.plot(r_bvp, S/(OMEGA/(-ALPHA)**0.5), label=r"$S$", linewidth=2.0)[0]
hp = axL.plot(r_bvp, 8*p_r, label=r"$8P_r$", linewidth=2.0, color = 'yellow')[0]  # scaled by 8 for visibility
hSrr = axL.plot(r_bvp, S_rr, label=r"$8S_r^r$", linewidth=2.0, color='magenta')[0]
axL.axhline(0, lw=1, alpha=0.35)
axL.set_ylim(-40, 40)
axL.set_xlabel(r"$x$")
axL.set_ylabel(r"$B,\,B',\,B'',\,S$", color=hB.get_color())
axL.tick_params(axis="y", colors=hB.get_color())
prettify(axL);
axL.set_xlim(0, 10)
axL.set_ylim(-0.5, 3)

# One combined legend outside the axes
H1, N1 = axL.get_legend_handles_labels()
axL.legend(H1, N1, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)

safe_savefig(fig, "marshmallow_B_Bp_Bpp_S.pdf", bbox_inches="tight",
             metadata={"Title": "B, B', B'' and S"})
safe_savefig(fig, "marshmallow_B_Bp_Bpp_S.png", dpi=300, bbox_inches="tight")

plt.show()



#
# # ================================ GRID PLOT BLOCK ==================================
# # Pairwise grid + bottom profiles, all on r_bvp
# # The 6×6 grid shows scatter relationships between each pair of variables
# # (w, m, gamma, kappa, S, y), colored by r. The bottom row shows profiles vs r.
# names  = ["w", "m", "gamma", "kappa", "S", "y"]
# arrays = [w,    m,   gamma,   kappa,   S,   y]
#
# # Print values at the first mesh point (proxy for origin)
# origin_vals = []
# for i in range(4):
#     print(names[i], "=", arrays[i][0], "at the origin-ish mesh point")
#     origin_vals.append(arrays[i][0])
#
# # Derived curvature at the first mesh point (using the same formula as above).
# R_origin = 2*(origin_vals[0] - 1) / r_bvp[0]**2 + 6 * origin_vals[2] / r_bvp[0] - 12 * origin_vals[3]
# print("R at the origin-ish mesh point is", R_origin)
#
# # Layout: 6×6 grid above + bottom row of profiles
# fig = plt.figure(figsize=(18, 21))
# gs  = fig.add_gridspec(nrows=7, ncols=6, height_ratios=[1]*6 + [1.1])
#
# # 6×6 pairwise grid (rows 0..5)
# axes = [[fig.add_subplot(gs[i, j]) for j in range(6)] for i in range(6)]
# for i in range(6):
#     for j in range(6):
#         ax = axes[i][j]
#         if i == j:
#             # Diagonal cells: label with the variable name.
#             ax.text(0.5, 0.5, names[i], ha="center", va="center", fontsize=12, weight="bold")
#             ax.set_xticks([]); ax.set_yticks([])
#         elif i < j:
#             # Upper triangle: left intentionally blank to avoid duplication.
#             ax.set_xticks([]); ax.set_yticks([])
#         else:
#             # Lower triangle: scatter plot of (var_j, var_i), colored by r.
#             xi = arrays[j]; yi = arrays[i]
#             valid = np.isfinite(xi) & np.isfinite(yi)
#             ax.scatter(xi[valid], yi[valid], c=r_bvp[valid], s=5, cmap="viridis")
#             if i == 5:  # bottom row of the grid → x-labels
#                 ax.set_xlabel(names[j])
#             if j == 0:  # first column → y-labels
#                 ax.set_ylabel(names[i])
#             ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
#             ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
#
# # Bottom profiles (row index 6): w(r), m(r), gamma(r), kappa(r), S(r), y(r)
# labels_bottom = [r"$w(r)$", r"$m(r)$", r"$\gamma(r)$", r"$\kappa(r)$", r"$S(r)$", r"$y(r)$"]
# for j, (lab, arr) in enumerate(zip(labels_bottom, arrays)):
#     ax = fig.add_subplot(gs[6, j])
#     ax.plot(r_bvp, arr)
#     ax.set_title(lab)
#     ax.set_xlabel(r"$r$")
#     ax.grid(True, alpha=0.3)
#     ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
#     ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
#
# plt.suptitle("All variable pairs (color = r) — evaluated on solve_bvp mesh", fontsize=16)
# plt.tight_layout(rect=[0, 0, 0.95, 0.95])
# safe_savefig(fig, "marshmallow_all_pairs_with_profiles.png", dpi=300, bbox_inches="tight")
# plt.show()

print("S_0 (first mesh point) = ", S[0])
print("final value of S",S[-1])
print(S_rr[0])


# ====================== MULTI-LAMBDA OVERLAY: B, B', B'', S =======================
# Configure which bare λ values you want to compare (before the MK_ON rescale).
LAMBDAS_SWEEP = [0.25*1000, 0.25*100, 0.25*10, 0.25*1]

def solve_and_profiles_for_lambda(lambda_bare):
    """
    Re-solve the BVP for a given bare lambda and return (r, B, Bp, Bpp, S).
    Respects your MK_ON rescaling convention.
    """
    global LAMBDA  # rhs() and bc() read this
    LAMBDA_saved = LAMBDA

    # Apply the same MK_ON rescaling you do at the top of the script
    LAMBDA_eff = (lambda_bare/4.0) if MK_ON else lambda_bare
    LAMBDA = LAMBDA_eff

    # Fresh mesh + fresh initial guess (depend on LAMBDA via S_inf)
    rmin_loc, rmax_loc = 1e-6, RMAX
    r_init_loc = np.geomspace(rmin_loc, rmax_loc, NPTS)

    Y_guess_loc = np.zeros((6, r_init_loc.size))
    Y_guess_loc[0] = 1.0                 # w
    Y_guess_loc[2] = 0.0                 # gamma
    Y_guess_loc[3] = KAPPA_INF           # kappa
    S_inf_loc = (np.abs(-SIGMA * KAPPA_INF / (2.0 * LAMBDA)))**0.5
    Y_guess_loc[4] = S_inf_loc           # S
    Y_guess_loc[5] = 0.0                 # y

    sol_loc = solve_bvp(rhs, bc, r_init_loc, Y_guess_loc,
                        tol=3e-9, max_nodes=200000, verbose=0)

    if sol_loc.status != 0:
        print(f"[\lambda={lambda_bare}] solve_bvp message:", sol_loc.message)

    r = sol_loc.x
    w_loc, m_loc, g_loc, k_loc, S_loc, y_loc = sol_loc.y
    rs_loc = np.where(r == 0.0, np.finfo(float).eps, r)

    # Metric + derivatives for this lambda
    B_loc   = w_loc - 2.0*m_loc/rs_loc + g_loc*rs_loc - k_loc*rs_loc**2
    Bp_loc  = 2.0*m_loc/rs_loc**2 + g_loc - 2.0*k_loc*rs_loc
    Bpp_loc = -4.0*m_loc/rs_loc**3 - 2.0*k_loc

    # Restore global LAMBDA so the rest of your script stays consistent
    LAMBDA = LAMBDA_saved
    return r, B_loc, Bp_loc, Bpp_loc, S_loc

# Solve for all requested lambdas
multi = []
for lam in LAMBDAS_SWEEP:
    try:
        multi.append((lam, *solve_and_profiles_for_lambda(lam)))
    except Exception as e:
        print(f"Failed for \lambda={lam}: {e}")

# ---------- Plots: 2×2 grid with B, B', B'', S (overlay curves per λ) -------------
fig, axs = plt.subplots(2, 2, constrained_layout=True, figsize=(11, 8.5))
(axB, axBp), (axBpp, axS) = axs

# Stable color per lambda across all subplots
palette = list(plt.cm.tab10.colors)  # colorblind-friendly, repeats if >10 curves
lambda_colors = {}                   # map λ -> color

for i, (lam, r, Bc, Bpc, Bppc, Sc) in enumerate(multi):
    c = palette[i % len(palette)]
    lambda_colors[lam] = c

    # plot each quantity with the SAME color for this λ
    axB.plot(r,   Bc,    color=c, lw=2.0)
    axBp.plot(r,  Bpc,   color=c, lw=2.0)
    axBpp.plot(r, Bppc,  color=c, lw=2.0)
    axS.plot(r,   Sc,    color=c, lw=2.0)

# Cosmetics / axes (apply r ≤ 10 only here)
for ax in [axB, axBp, axBpp, axS]:
    ax.axhline(0, lw=1, alpha=0.35)
    prettify(ax)
    ax.set_xlim(0, 10)
    ax.set_xlabel(r"$r$")

axB.set_title(r"$B(r)$")
axB.set_ylim(-0.5,3)
axBp.set_title(r"$B'(r)$")
axBp.set_ylim(-0.5,3)
axBpp.set_title(r"$B''(r)$")
axS.set_title(r"$S(r)$")

# ----- Color legend for lambdas (one shared legend for the whole figure) -----
from matplotlib.lines import Line2D
lambda_handles = [
    Line2D([0], [0], color=lambda_colors[lam], lw=3, label=rf"$\lambda={lam:g}$")
    for lam, *_ in multi
]
fig.legend(lambda_handles, [h.get_label() for h in lambda_handles],
           loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False,
           title=r"Color = $\lambda$")

safe_savefig(fig, "marshmallow_multi_lambda_B_Bp_Bpp_S.pdf", bbox_inches="tight",
             metadata={"Title": "B, B', B'' and S vs r for multiple \lambda (color-coded by \lambda)"})
safe_savefig(fig, "marshmallow_multi_lambda_B_Bp_Bpp_S.png", dpi=300, bbox_inches="tight")

plt.show()

# ====================== MULTI-λ: VELOCITY PROFILES (log–log) ======================
# This block assumes LAMBDAS_SWEEP, EPS_B, MK_ON, KAPPA_INF, SIGMA, LAMBDA, rhs, bc,
# prettify, and safe_savefig are already defined above.

# --- Helper: per-λ solve that RETURNS y so we can use analytic S' ---
def solve_and_profiles_for_lambda(lambda_bare):
    global LAMBDA
    LAMBDA_saved = LAMBDA
    LAMBDA_eff = (lambda_bare/4.0) if MK_ON else lambda_bare
    LAMBDA = LAMBDA_eff

    rmin_loc, rmax_loc = 1e-6, RMAX
    r_init_loc = np.geomspace(rmin_loc, rmax_loc, NPTS)

    Y_guess_loc = np.zeros((6, r_init_loc.size))
    Y_guess_loc[0] = 1.0                 # w
    Y_guess_loc[2] = 0.0                 # gamma
    Y_guess_loc[3] = KAPPA_INF           # kappa
    S_inf_loc = (np.abs(-SIGMA * KAPPA_INF / (2.0 * LAMBDA)))**0.5
    Y_guess_loc[4] = S_inf_loc           # S
    Y_guess_loc[5] = 0.0                 # y

    sol_loc = solve_bvp(rhs, bc, r_init_loc, Y_guess_loc,
                        tol=3e-9, max_nodes=200000, verbose=0)
    if sol_loc.status != 0:
        print(f"[\\lambda={lambda_bare}] solve_bvp message:", sol_loc.message)

    r = sol_loc.x
    w_loc, m_loc, g_loc, k_loc, S_loc, y_loc = sol_loc.y
    rs = np.where(r == 0.0, np.finfo(float).eps, r)
    B    = w_loc - 2.0*m_loc/rs + g_loc*rs - k_loc*rs**2
    Bp   = 2.0*m_loc/rs**2 + g_loc - 2.0*k_loc*rs
    Bpp  = -4.0*m_loc/rs**3 - 2.0*k_loc

    LAMBDA = LAMBDA_saved
    return r, B, Bp, Bpp, S_loc, y_loc

# --- Build fresh solutions (include y) ---
multi = []
for lam in LAMBDAS_SWEEP:
    try:
        multi.append((lam, *solve_and_profiles_for_lambda(lam)))
    except Exception as e:
        print(f"Failed for \\lambda={lam}: {e}")

# --- Observational data (MW rotation curve) ---
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
c = 2.99792458e10
kmps_to_kpcps = 1 / (c / 1e5)
y_values = [y * kmps_to_kpcps for y in y_values_kmps]
mask_data = (np.array(x_values) > 0) & (np.array(y_values) > 0)
xd = np.array(x_values)[mask_data]
yd = np.array(y_values)[mask_data]

# --- Velocity from profiles using analytic S' and safe masks ---
EPS_B_local = max(EPS_B, 1e-12)
R_CUT = 1e-3   # inner cutoff for circular-orbit diagnostic

def velocity_from_profiles(r, B, Bp, S, y):
    r = np.asarray(r); B = np.asarray(B); Bp = np.asarray(Bp)
    S = np.asarray(S); y = np.asarray(y)

    rsafe = np.where(r == 0.0, np.finfo(float).eps, r)
    Bsafe = np.sign(np.where(B == 0.0, 1.0, B)) * np.maximum(np.abs(B), EPS_B_local)
    Sp = y / (rsafe**2 * Bsafe)  # analytic S'

    Sfloor = 1e-10 * max(1.0, np.nanmax(np.abs(S)))
    Ssafe = np.where(np.abs(S) > Sfloor, S, np.nan)

    vB2 = r * Bp / (2.0 * Bsafe)
    vS2 = r * Sp / Ssafe
    den = 1.0 + vS2
    num = vB2 + vS2

    valid = (
        (r >= R_CUT) &
        np.isfinite(num) & np.isfinite(den) &
        (den > 0.0) & (num >= 0.0) &
        (Bsafe > 0.0)
    )
    v2 = np.empty_like(r); v2[:] = np.nan
    v2[valid] = num[valid] / den[valid]
    m = valid & (v2 > 0.0)
    return r[m], np.sqrt(v2[m])

# --- Plot: log10 r vs log10 v, with an INSIDE legend mapping color→λ ---
fig, ax = plt.subplots(constrained_layout=True, figsize=(7.8, 6.2))

# consistent colors per λ
palette = list(plt.cm.tab10.colors)
lambda_colors = {lam: palette[i % len(palette)] for i, (lam, *_) in enumerate(multi)}

# curves
for (lam, r, Bc, Bpc, Bppc, Sc, yc) in multi:
    rm, vm = velocity_from_profiles(r, Bc, Bpc, Sc, yc)
    if rm.size:
        ax.plot(np.log10(rm), np.log10(vm), lw=2.0, color=lambda_colors[lam])

# MW data
ax.scatter(np.log10(xd), np.log10(yd), s=28, facecolors='none',
           edgecolors='crimson', label="MW")

ax.set_xlabel(r"$\log_{10} r$")
ax.set_ylabel(r"$\log_{10} v$")
prettify(ax)

# Build legend entries INSIDE the axes to avoid clipping on save
from matplotlib.lines import Line2D
plotted_lams = [lam for (lam, *_) in multi if lam in lambda_colors]
lambda_handles = [
    Line2D([0], [0], color=lambda_colors[lam], lw=3, label=rf"$\lambda={lam:g}$")
    for lam in plotted_lams
]
mw_handle = Line2D([0], [0], marker='o', linestyle='none', markersize=6,
                   markerfacecolor='white', markeredgecolor='crimson', label="MW")

leg = ax.legend(
    lambda_handles + [mw_handle],
    [h.get_label() for h in lambda_handles] + ["MW"],
    loc="upper left", bbox_to_anchor=(0.02, 0.98),
    frameon=True, fancybox=True, framealpha=0.85,
    title=r"Color = $\lambda$",
    ncol=1
)

safe_savefig(fig, "marshmallow_v_loglog_multi_lambda.pdf", bbox_inches="tight")
safe_savefig(fig, "marshmallow_v_loglog_multi_lambda.png", dpi=300, bbox_inches="tight")
plt.show()
