"""
Fly-around trajectory generator — Python port of flyAroundTrajGen.m.

Convex (SOCP) min-fuel trajectory from x0 to xf in a fixed time of flight,
with a spherical keep-out zone (KOZ) centered at the origin. Dynamics are
Clohessy-Wiltshire about a chief circular orbit. The non-convex exclusion
||r_k|| >= rKOZ is handled by sequential convex programming: at each
iteration, nodes that violate the KOZ contribute tangent-plane cuts
n_k' * r_k >= rKOZ  (with n_k = r_k / ||r_k||), which are valid outer
approximations of the sphere. Cuts are accumulated across iterations, so
the feasible set tightens monotonically and the loop terminates as soon
as the current iterate is both KOZ-feasible and stationary.

Requires: numpy, cvxpy, and a cvxpy-compatible SOCP solver (MOSEK by default).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cvxpy as cp
import numpy as np


@dataclass
class FlyAroundOpts:
    n: float                                  # chief mean motion [rad/s]
    rKOZ: float                               # KOZ sphere radius
    alim: float                               # per-axis accel magnitude limit
    dt: float                                 # node spacing (time)
    max_iter: int = 20                        # max SCP iterations
    tol: float = 1.0                          # position convergence tol
    verbose: bool = False
    solver: str = "MOSEK"                     # any cvxpy SOCP solver
    solver_opts: dict[str, Any] = field(default_factory=dict)


def cw_stm(n: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """CW state-transition (A) and control-effect (B) matrices for step dt."""
    c, s = np.cos(n * dt), np.sin(n * dt)
    Phi_rr = np.array([
        [4 - 3 * c,       0, 0],
        [6 * (s - n * dt), 1, 0],
        [0,                0, c],
    ])
    Phi_rv = np.array([
        [s / n,              2 * (1 - c) / n,           0],
        [2 * (c - 1) / n,    (4 * s - 3 * n * dt) / n,  0],
        [0,                  0,                         s / n],
    ])
    Phi_vr = np.array([
        [3 * n * s,      0, 0],
        [6 * n * (c - 1), 0, 0],
        [0,              0, -n * s],
    ])
    Phi_vv = np.array([
        [c,     2 * s,       0],
        [-2 * s, 4 * c - 3,  0],
        [0,     0,           c],
    ])
    A = np.block([[Phi_rr, Phi_rv], [Phi_vr, Phi_vv]])
    B = np.vstack([Phi_rv, Phi_vv])
    return A, B


def solve_cw_koz(
    x0: np.ndarray,
    xf: np.ndarray,
    tvec: np.ndarray,
    opts: FlyAroundOpts,
    cut_idx: np.ndarray | None = None,
    cut_normals: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, float, bool, str]:
    """Single SOCP solve: CW dynamics, min sum ||u_i||_2, optional KOZ cuts.

    Parameters
    ----------
    x0, xf      : (6,) initial / terminal relative states [r; v].
    tvec        : (N+1,) node times, strictly increasing, tvec[0] = 0.
    opts        : FlyAroundOpts.
    cut_idx     : (K,) node indices to constrain (0-based).
    cut_normals : (K, 3) unit outward normals for the tangent-plane cuts.

    Returns
    -------
    traj     : (N+1, 6) or None if infeasible
    dvs      : (N+1, 3) with last row zero, or None
    cost     : sum of ||u_i||_2 (np.inf if infeasible)
    feasible : True if solver returned optimal / optimal_inaccurate
    status   : cvxpy problem status string
    """
    if cut_idx is None:
        cut_idx = np.zeros(0, dtype=int)
    if cut_normals is None:
        cut_normals = np.zeros((0, 3))

    x0 = np.asarray(x0, dtype=float).reshape(-1)
    xf = np.asarray(xf, dtype=float).reshape(-1)
    tvec = np.asarray(tvec, dtype=float).reshape(-1)

    N = len(tvec) - 1
    dts = np.diff(tvec)
    dvlim = opts.alim * dts

    X = cp.Variable((N + 1, 6), name="X")
    U = cp.Variable((N, 3), name="U")

    constraints = [X[0, :] == x0, X[-1, :] == xf]

    for i in range(N):
        A, B = cw_stm(opts.n, dts[i])
        constraints.append(X[i + 1, :] == A @ X[i, :] + B @ U[i, :])
        constraints.append(cp.norm(U[i, :], 2) <= dvlim[i])

    for j in range(len(cut_idx)):
        k = int(cut_idx[j])
        constraints.append(cut_normals[j, :] @ X[k, :3] >= opts.rKOZ)

    objective = cp.Minimize(cp.sum(cp.norm(U, 2, axis=1)))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=opts.solver, **opts.solver_opts)

    status = problem.status
    feasible = status in ("optimal", "optimal_inaccurate")
    if feasible and X.value is not None:
        traj = np.asarray(X.value)
        ctrls = np.asarray(U.value)
        dvs = np.vstack([ctrls, np.zeros((1, 3))])
        cost = float(problem.value)
        return traj, dvs, cost, True, status
    return None, None, float("inf"), False, status


def fly_around_traj_gen(
    x0: np.ndarray,
    xf: np.ndarray,
    TOF: float,
    opts: FlyAroundOpts,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Convex min-fuel trajectory from x0 to xf in time TOF, avoiding a
    spherical KOZ centered at the origin.

    Returns
    -------
    traj : (N+1, 6) state history at node times.
    dvs  : (N+1, 3) delta-V per node (last row forced to 0).
    info : dict with keys
             'tvec', 'iters', 'cost' (list), 'n_cuts' (list),
             'min_rad' (list), 'feasible' (bool), 'status' (last solver status).
    """
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    xf = np.asarray(xf, dtype=float).reshape(-1)

    if np.linalg.norm(x0[:3]) <= opts.rKOZ:
        raise ValueError("x0 lies inside the KOZ.")
    if np.linalg.norm(xf[:3]) <= opts.rKOZ:
        raise ValueError("xf lies inside the KOZ.")

    tvec = np.unique(np.concatenate([np.arange(0.0, TOF, opts.dt), [TOF]]))
    N = len(tvec) - 1

    # Straight-line state guess (positions and velocities interpolated).
    tfrac = tvec / TOF
    xprev = x0[None, :] + (xf - x0)[None, :] * tfrac[:, None]  # (N+1, 6)

    cut_idx = np.zeros(0, dtype=int)
    cut_normals = np.zeros((0, 3))

    info: dict[str, Any] = {
        "tvec": tvec,
        "iters": 0,
        "cost": [],
        "n_cuts": [],
        "min_rad": [],
        "feasible": False,
        "status": "",
    }

    traj = xprev.copy()
    dvs = np.zeros((N + 1, 3))
    pos_tol = 1e-6 * max(1.0, opts.rKOZ)

    for it in range(opts.max_iter):
        info["iters"] = it + 1

        rn = np.linalg.norm(xprev[:, :3], axis=1)
        viol = np.where(rn < opts.rKOZ - pos_tol)[0]

        # Skip violators sitting essentially at the origin — their radial
        # direction is ill-defined and a zero normal gives an infeasible cut.
        if len(viol) > 0:
            rn_v = rn[viol]
            keep = rn_v > 1e-6 * opts.rKOZ
            if np.any(keep):
                v_keep = viol[keep]
                new_normals = xprev[v_keep, :3] / rn_v[keep, None]
                cut_idx = np.concatenate([cut_idx, v_keep])
                cut_normals = np.vstack([cut_normals, new_normals])

        traj_new, dvs_new, cost, feas, status = solve_cw_koz(
            x0, xf, tvec, opts, cut_idx, cut_normals
        )
        info["status"] = status

        if not feas:
            if opts.verbose:
                print(
                    f"[fly_around_traj_gen] iter {it + 1}: SOCP infeasible ({status})."
                )
            info["feasible"] = False
            return traj, dvs, info

        rn_new = np.linalg.norm(traj_new[:, :3], axis=1)
        min_rad = float(np.min(rn_new))
        delta = float(
            np.sum(np.linalg.norm(traj_new[:, :3] - xprev[:, :3], axis=1))
        )
        koz_feas = min_rad >= opts.rKOZ - pos_tol

        info["cost"].append(cost)
        info["n_cuts"].append(cut_normals.shape[0])
        info["min_rad"].append(min_rad)

        if opts.verbose:
            print(
                f"[fly_around_traj_gen] iter {it + 1:2d}  "
                f"new_viol={len(viol):3d}  cuts={cut_normals.shape[0]:3d}  "
                f"cost={cost:.4g}  delta={delta:.4g}  min|r|={min_rad:.3f}"
            )

        traj = traj_new
        dvs = dvs_new
        xprev = traj_new

        if koz_feas and delta < opts.tol:
            info["feasible"] = True
            return traj, dvs, info

    rn_final = np.linalg.norm(traj[:, :3], axis=1)
    info["feasible"] = bool(float(np.min(rn_final)) >= opts.rKOZ - pos_tol)
    return traj, dvs, info
