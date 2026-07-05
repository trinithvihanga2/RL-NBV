"""
Clohessy-Wiltshire (CW) Dynamics & Trajectory Planner Utility
=============================================================

What is this?
-------------
When a satellite (or camera drone) needs to move from one viewpoint to another
while orbiting a target object, it must fire its thrusters.

The Clohessy-Wiltshire equations model this motion in a *relative* frame —
that is, the position of the spacecraft relative to a reference point that is
already in a circular orbit around the target.

This module provides:
  - CWDynamics class : A wrapper around a Sequential Convex Programming (SCP)
                       trajectory planner that computes the minimum-fuel delta-v
                       (Δv) needed to travel between two orbital positions while
                       avoiding a spherical Keep-Out Zone (KOZ).

Key concept — what is Δv?
--------------------------
Δv (delta-v) is the total change in velocity needed to make a manoeuvre.
It is the standard "fuel cost" in astrodynamics.

The Trajectory Planner (SCP)
--------------------------------------
Instead of a simple 2-impulse direct transfer (which might cut through the
target object), this module integrates the `fly_around_traj_gen` module.
It sets up a convex optimization problem using CVXPY to find a multi-waypoint
trajectory that minimizes the sum of velocity impulses (Δv) while strictly
respecting dynamic constraints and the KOZ boundary.

Units
-----
All quantities here are dimensionless (unit-sphere coordinates).
Viewpoints on the unit sphere are scaled to orbit_radius before computation
so that the relative position vectors have physically correct magnitudes.
"""

import numpy as np
import logging

from .fly_around_traj_gen import fly_around_traj_gen, FlyAroundOpts

logger = logging.getLogger(__name__)

# Condition-number threshold above which the position-control block is singular.
_COND_THRESHOLD = 1e10


# =============================================================================
# State-Transition Matrix (STM)
# =============================================================================


def _build_state_control_matrices(n: float, t: float):
    """
    Build fly-around-style CW discrete dynamics matrices.

    Returns
    -------
    A : np.ndarray, shape (6, 6)
        Coasting state transition so that x_{k+1} = A x_k + B u_k.
    B : np.ndarray, shape (6, 3)
        Control-effect matrix where u_k is an impulsive Δv at node k.
    """
    nt = n * t
    s = np.sin(nt)
    c = np.cos(nt)

    Phi_rr = np.array(
        [
            [4 - 3 * c, 0, 0],
            [6 * (s - nt), 1, 0],
            [0, 0, c],
        ]
    )
    Phi_rv = np.array(
        [
            [s / n, 2 * (1 - c) / n, 0],
            [2 * (c - 1) / n, (4 * s - 3 * nt) / n, 0],
            [0, 0, s / n],
        ]
    )
    Phi_vr = np.array(
        [
            [3 * n * s, 0, 0],
            [6 * n * (c - 1), 0, 0],
            [0, 0, -n * s],
        ]
    )
    Phi_vv = np.array(
        [
            [c, 2 * s, 0],
            [-2 * s, 4 * c - 3, 0],
            [0, 0, c],
        ]
    )

    A = np.block([[Phi_rr, Phi_rv], [Phi_vr, Phi_vv]])
    B = np.vstack([Phi_rv, Phi_vv])
    return A, B


def _build_stm(n: float, t: float):
    """
    Build the four 3×3 blocks of the CW state-transition matrix.

    The STM maps the *initial* relative state (position r0, velocity v0) to
    the *final* relative state (position rf, velocity vf) after time t.

    Parameters
    ----------
    n : float
        Mean orbital motion  n = √(μ / a³),  in rad/time-unit.
        For our unit-sphere setup with μ=1 and a=1, this equals 1.0.
    t : float
        Time of flight (dimensionless time units).  Must be > 0.

    Returns
    -------
    Phi_rr : np.ndarray, shape (3, 3)
        Position-to-position block.
    Phi_rv : np.ndarray, shape (3, 3)
        Velocity-to-position block.  We invert this to find v0.
    Phi_vr : np.ndarray, shape (3, 3)
        Position-to-velocity block.
    Phi_vv : np.ndarray, shape (3, 3)
        Velocity-to-velocity block.

    Notes
    -----
    The coordinate axes follow the Hill (LVLH) frame:
      x : radial   (away from the central body)
      y : along-track (tangential, direction of orbital motion)
      z : cross-track (out-of-plane, normal to orbit)
    """
    A, _ = _build_state_control_matrices(n, t)
    Phi_rr = A[:3, :3]
    Phi_rv = A[:3, 3:]
    Phi_vr = A[3:, :3]
    Phi_vv = A[3:, 3:]

    return Phi_rr, Phi_rv, Phi_vr, Phi_vv


# =============================================================================
# CWDynamics  –  the main class
# =============================================================================


class CWDynamics:
    """
    Clohessy-Wiltshire rendezvous dynamics.

    Usage
    -----
    ::

        cw = CWDynamics(mean_motion=1.0)

        delta_v, v0, vf = cw.compute_delta_v(
            r0 = viewpoints[3] * orbit_radius,   # scale from unit sphere
            rf = viewpoints[7] * orbit_radius,
            t  = travel_times[3, 7],             # time of flight
        )

        print(f"Fuel cost: {delta_v:.4f}")

    Parameters
    ----------
    mean_motion : float
        Orbital mean motion n [rad / time-unit].
        For a circular orbit of radius *a* with gravitational parameter μ:
            n = sqrt(μ / a³)
        With our defaults (μ=1, a=1) this is 1.0.
    """

    def __init__(self, mean_motion: float, scp_config: dict = {
            "koz_radius": 0.95,
            "alim": 1000.0,
            "dt": 0.5,
            "max_iter": 20,
            "solver": "ECOS"
        }):
        self.n = mean_motion
        self.scp_config = scp_config
        self.last_trajectory = None

    # -------------------------------------------------------------------------
    def compute_delta_v(
        self,
        r0: np.ndarray,
        rf: np.ndarray,
        t: float,
    ):
        """
        Compute the minimum-fuel Δv required to travel from r0 to rf in time t.

        Algorithm (SCP Fly-Around)
        --------------------------
        This function uses Sequential Convex Programming (SCP) to generate a
        collision-free trajectory between r0 and rf around a Keep-Out Zone (KOZ).
        
        1. Formulates the boundary states: x0 = [r0, 0, 0, 0] and xf = [rf, 0, 0, 0].
        2. Configures FlyAroundOpts with the selected convex solver (e.g. ECOS).
        3. Calls `fly_around_traj_gen` to solve for the optimal multi-waypoint path.
        4. Calculates the total Δv as the sum of L2 norms of all control impulses along the path.

        Parameters
        ----------
        r0 : np.ndarray, shape (3,)
            Initial relative position (already scaled to orbit radius).
        rf : np.ndarray, shape (3,)
            Final relative position (already scaled to orbit radius).
        t  : float
            Time of flight. Should be > 0; if 0 the spacecraft is already
            at the destination and Δv = 0 by definition.

        Returns
        -------
        delta_v : float
            Total Δv (sum of impulse norms). Returns np.inf if the manoeuvre is infeasible.
        v0 : np.ndarray or None
            The first control impulse vector from the generated trajectory.
        vf : np.ndarray or None
            The final control impulse vector from the generated trajectory.
        """
        # Trivial case: no movement needed
        if t <= 0.0:
            self.last_trajectory = np.array([r0, rf])
            return 0.0, np.zeros(3), np.zeros(3)

        # Construct FlyAroundOpts
        opts = FlyAroundOpts(
            n=self.n,
            rKOZ=float(self.scp_config.get("koz_radius", 0.95)),
            alim=float(self.scp_config.get("alim", 1000.0)),
            dt=float(self.scp_config.get("dt", 0.5)),
            max_iter=int(self.scp_config.get("max_iter", 20)),
            solver=self.scp_config.get("solver", "ECOS")
        )
        
        # Prepare start and end states [r, v] where v=0
        x0 = np.concatenate([r0, np.zeros(3)])
        xf = np.concatenate([rf, np.zeros(3)])
        
        # Suppress verbose output
        opts.verbose = False
        
        try:
            traj, dvs, info = fly_around_traj_gen(x0, xf, t, opts)
            if not info.get("feasible", False):
                logger.debug(f"SCP planner failed at t={t:.4f}")
                self.last_trajectory = None
                return np.inf, None, None
                
            delta_v = float(np.sum(np.linalg.norm(dvs, axis=1)))
            self.last_trajectory = traj[:, :3]
            return delta_v, dvs[0], dvs[-1]
        except Exception as e:
            logger.error(f"SCP planner exception: {e}")
            self.last_trajectory = None
            return np.inf, None, None

    # -------------------------------------------------------------------------
    def compute_final_velocity(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        t: float,
    ):
        """
        Propagate (r0, v0) forward by time t and return the final state.

        Useful for visualising the trajectory after solving for v0.

        Parameters
        ----------
        r0 : np.ndarray, shape (3,)   Initial relative position.
        v0 : np.ndarray, shape (3,)   Initial relative velocity.
        t  : float                    Time of flight.

        Returns
        -------
        rf : np.ndarray, shape (3,)   Final relative position.
        vf : np.ndarray, shape (3,)   Final relative velocity.
        """
        A, _ = _build_state_control_matrices(self.n, t)
        x0 = np.concatenate([r0, v0])
        xf = A @ x0
        rf = xf[:3]
        vf = xf[3:]
        return rf, vf

# =============================================================================
# Quick self-test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("CW Dynamics — Quick Self-Test")
    print("=" * 60)

    n = 1.0  # mean motion (unit-sphere defaults)
    cw = CWDynamics(mean_motion=n)

    # Two viewpoints on the unit sphere (already on surface)
    r0 = np.array([1.0, 0.0, 0.0])  # "front"
    rf = np.array([0.0, 1.0, 0.0])  # "left"

    # Time of flight: quarter orbit
    t = (2.0 * np.pi / n) / 4.0  # ≈ π/2

    dv, v0, vf = cw.compute_delta_v(r0, rf, t)
    rf_check, _ = cw.compute_final_velocity(r0, v0, t)

    print(f"\nInitial position   : {r0}")
    print(f"Target  position   : {rf}")
    print(f"Time of flight     : {t:.4f} time-units")
    print(f"Departure velocity : {v0}")
    print(f"Arrival  velocity  : {vf}")
    print(f"Δv departure       : {np.linalg.norm(v0):.6f}")
    print(f"Δv arrival (brake) : {np.linalg.norm(vf):.6f}")
    print(f"Δv total           : {dv:.6f}")
    print(f"Propagated final   : {rf_check}   (should match target)")
    print(f"Position error     : {np.linalg.norm(rf_check - rf):.2e}   (should be ~0)")

    # ── Singularity test ────────────────────────────────────────────────────
    print("\n--- Singularity test (t = π/n, i.e. half orbital period) ---")
    t_singular = np.pi / n
    dv_s, v0_s, vf_s = cw.compute_delta_v(r0, rf, t_singular)
    print(f"Δv at t=π/n : {dv_s}   (should be inf)")
