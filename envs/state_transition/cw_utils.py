"""
Clohessy-Wiltshire (CW) Dynamics Utility
==========================================

What is this?
-------------
When a satellite (or camera drone) needs to move from one viewpoint to another
while orbiting a target object, it must fire its thrusters.

The Clohessy-Wiltshire equations model this motion in a *relative* frame —
that is, the position of the spacecraft relative to a reference point that is
already in a circular orbit around the target.

This module provides:
  - CWDynamics class : computes the delta-v (Δv) needed to travel between two
                       points in a given time.
  - compute_delta_v_matrix() : precomputes Δv for every pair of the 33
                                viewpoints so the RL environment can do O(1)
                                lookups during training.

Key concept — what is Δv?
--------------------------
Δv (delta-v) is the total change in velocity needed to make a manoeuvre.
It is the standard "fuel cost" in astrodynamics:
  - More Δv  →  more fuel burned  →  more expensive manoeuvre
  - Less Δv  →  cheaper, more fuel-efficient path

The CW state-transition matrix (STM)
--------------------------------------
Given:
  r0  : initial relative position  (3-vector, in orbital-radius units)
  v0  : initial relative velocity  (3-vector, we solve for this)
  rf  : desired final position      (3-vector)
  t   : time of flight              (scalar)

The CW equations relate (r0, v0) → (rf, vf) linearly:

  rf = Φ_rr(t) · r0  +  Φ_rv(t) · v0    … (position equation)
  vf = Φ_vr(t) · r0  +  Φ_vv(t) · v0    … (velocity equation)

We *know* r0 and rf.  We *want* to find v0.
Rearranging the position equation:

  Φ_rv(t) · v0 = rf − Φ_rr(t) · r0

This is a 3×3 linear system.  Solving it gives v0.
The total Δv = ‖v0‖ + ‖vf‖ (sum of initial and final velocity magnitudes).

Units
-----
All quantities here are dimensionless (unit-sphere coordinates).
Viewpoints on the unit sphere are scaled to orbit_radius before CW computation
so that the relative position vectors have physically correct magnitudes.
"""

import numpy as np
import logging

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

    def __init__(self, mean_motion: float):
        self.n = mean_motion

    # -------------------------------------------------------------------------
    def compute_delta_v(
        self,
        r0: np.ndarray,
        rf: np.ndarray,
        t: float,
    ):
        """
        Compute the Δv required to travel from r0 to rf in time t.

        Algorithm (fly-around compatible)
        ---------------------------------
        1.  Build discrete CW matrices A, B such that
            x1 = A x0 + B u0,
            where x = [r; v] and u0 is an impulsive departure Δv.
        2.  Assume current relative velocity is zero (x0 = [r0; 0]).
        3.  Solve position endpoint constraint for u0 using the top block:
            B_r · u0 = rf − A_rr · r0
        4.  Propagate final velocity with x1 and use braking burn vf.
        5.  Return Δv = ‖u0‖ + ‖vf‖.

        Parameters
        ----------
        r0 : np.ndarray, shape (3,)
            Initial relative position (already scaled to orbit radius).
        rf : np.ndarray, shape (3,)
            Final relative position (already scaled to orbit radius).
        t  : float
            Time of flight.  Should be > 0; if 0 the spacecraft is already
            at the destination and Δv = 0 by definition.

        Returns
        -------
        delta_v : float
            Total Δv = ‖v0‖ + ‖vf‖ (same units as r / time).
            Returns np.inf if the manoeuvre is dynamically infeasible (e.g.
            Φ_rv is singular at certain multiples of the orbital period).
        v0 : np.ndarray or None
            Required departure impulse vector u0. Returns None if infeasible.
            When infeasible, the tuple returned is (np.inf, None, None).
        """
        # Trivial case: no movement needed
        if t <= 0.0:
            return 0.0, np.zeros(3), np.zeros(3)

        A, B = _build_state_control_matrices(self.n, t)
        A_rr = A[:3, :3]
        B_r = B[:3, :]

        # Guard against singular position-control map (occurs at t = k·π/n, i.e. every
        # half orbital period — NOT every full period as is sometimes assumed).
        if np.linalg.cond(B_r) > _COND_THRESHOLD:
            logger.debug(
                f"CW singular at t={t:.4f}  (likely t ≈ k·π/n). Returning Δv = inf."
            )
            return np.inf, None, None

        # Endpoint equation from x1 = A x0 + B u0 with x0 = [r0; 0].
        rhs = rf - A_rr @ r0

        try:
            v0 = np.linalg.solve(B_r, rhs)
        except np.linalg.LinAlgError:
            logger.debug(f"CW solve failed at t={t:.4f}. Returning Δv = inf.")
            return np.inf, None, None

        # Propagate to final state and use final velocity magnitude as braking burn.
        x0 = np.concatenate([r0, np.zeros(3)])
        xf = A @ x0 + B @ v0
        vf = xf[3:]

        delta_v = float(np.linalg.norm(v0)) + float(np.linalg.norm(vf))
        return delta_v, v0, vf

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
# Batch pre-computation
# =============================================================================


def compute_delta_v_matrix(
    viewpoints: np.ndarray,
    travel_times: np.ndarray,
    orbit_radius: float,
    mean_motion: float,
) -> np.ndarray:
    """
    Pre-compute the Δv cost for every ordered pair (i → j) of viewpoints.

    This builds a square matrix so the RL environment can look up fuel costs
    in O(1) during training instead of solving a linear system every step.

    Parameters
    ----------
    viewpoints : np.ndarray, shape (N, 3)
        Unit-sphere viewpoints (normalized, ‖p‖ ≈ 1).
    travel_times : np.ndarray, shape (N, N)
        Pre-computed travel times.  Element [i, j] is the time of flight
        from viewpoint i to viewpoint j.
    orbit_radius : float
        Physical radius of the orbit (scales unit-sphere coords to real space).
    mean_motion : float
        Orbital mean motion n [rad / time-unit].

    Returns
    -------
    delta_v_matrix : np.ndarray, shape (N, N), dtype float32
        Element [i, j] = Δv needed to go from viewpoint i to viewpoint j.
        Diagonal entries are 0 (no movement).
        np.inf entries indicate dynamically infeasible transfers.

    Example
    -------
    ::

        dv_matrix = compute_delta_v_matrix(
            viewpoints   = vp,           # (33, 3)
            travel_times = tt,           # (33, 33)
            orbit_radius = 1.0,
            mean_motion  = 1.0,
        )
        fuel_cost = dv_matrix[current_view, next_view]
    """
    cw = CWDynamics(mean_motion)
    n = viewpoints.shape[0]

    delta_v_matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            if i == j:
                # Already at destination — no fuel needed
                delta_v_matrix[i, j] = 0.0
                continue

            # Scale unit-sphere coords to actual orbital radius
            r0 = viewpoints[i] * orbit_radius
            rf = viewpoints[j] * orbit_radius
            t = travel_times[i, j]

            dv_total, _, _ = cw.compute_delta_v(r0, rf, t)
            delta_v_matrix[i, j] = dv_total

    logger.info(
        f"Δv matrix computed: shape={delta_v_matrix.shape}, "
        f"min={delta_v_matrix[delta_v_matrix > 0].min():.4f}, "
        f"max={delta_v_matrix[delta_v_matrix < np.inf].max():.4f}, "
        f"mean={delta_v_matrix[delta_v_matrix > 0].mean():.4f}"
    )
    return delta_v_matrix


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
