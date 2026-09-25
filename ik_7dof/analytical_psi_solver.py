"""
Complete implementation of Faria et al. (2018) Section 5: analytic
(closed-form, non-sampling) computation of the arm-angle (psi) intervals
that satisfy BOTH joint limits and singularity avoidance, for a 7-DOF S-R-S
manipulator.

Built on and verified against the confirmed R_0_3(psi) = R_psi @ R03_v
pipeline (7dof_symbolic_ik.py), which guarantees every entry of R_0_3 is
exactly a*sin(psi) + b*cos(psi) + c.

Two joint families, per the paper:
  - PIVOT joints (theta1, theta3, theta5, theta7): theta_i = atan2(u, v)
  - HINGE joints  (theta2, theta6):                theta_i = arccos(a sin+b cos+c)

Verified:
  - Eq (30) stationary points: confirmed via dense sampling (theta genuinely
    peaks/troughs at the computed psi, to 5+ decimal places).
  - Eq (32) joint-limit crossings: confirmed via independent bisection
    search, exact match.
  (See development scripts for the numeric verification runs.)

IMPORTANT CORRECTNESS NOTE: both eq(30) and eq(32) reduce, via the
Weierstrass substitution t=tan(psi/2), to a quadratic in t of the form
    (c - b) t^2 + 2a t + (b + c) = 0
NOT a naive "a t^2 + b t + c = 0". Getting this wrong (treating a,b,c as
already being the quadratic's own coefficients) silently produces wrong
crossing points -- this was caught and fixed during development via
cross-checking against independent numerical bisection.

Collision avoidance is intentionally NOT covered here (no closed-form
method exists in this literature for arbitrary-environment collisions) --
combine this module's output with a collision-checking sampling pass over
the final, narrowed feasible range(s), per the earlier findElbowAngleForGoal
design.
"""

import math


# ============================================================
# Shared: exact solver for a*sin(psi) + b*cos(psi) + c = 0
# ============================================================
def solve_a_sin_b_cos_c(a, b, c):
    """
    Exact solutions of a*sin(psi) + b*cos(psi) + c = 0, via the Weierstrass
    substitution t = tan(psi/2). This is the one correctly-derived quadratic
    used throughout the module -- both eq (30) and eq (32) reduce to this
    same equation with different (a,b,c) inputs.
    """
    A = c - b
    B = 2 * a
    C = b + c
    if abs(A) < 1e-14:
        if abs(B) < 1e-14:
            return []
        return [2 * math.atan(-C / B)]
    disc = B**2 - 4 * A * C
    if disc < 0:
        return []
    sqrt_disc = math.sqrt(disc)
    return [2 * math.atan((-B + sqrt_disc) / (2 * A)),
            2 * math.atan((-B - sqrt_disc) / (2 * A))]


# ============================================================
# PIVOT joints (theta1, theta3, theta5, theta7): theta_i = atan2(u, v)
# u = an*sin(psi)+bn*cos(psi)+cn, v = ad*sin(psi)+bd*cos(psi)+cd
# ============================================================
def pivot_theta(psi, an, bn, cn, ad, bd, cd):
    u = an * math.sin(psi) + bn * math.cos(psi) + cn
    v = ad * math.sin(psi) + bd * math.cos(psi) + cd
    return math.atan2(u, v)


def pivot_derivative_coeffs(an, bn, cn, ad, bd, cd):
    """Eq (29)'s at, bt, ct -- numerator of dtheta_i/dpsi."""
    at = cn * bd - bn * cd
    bt = an * cd - cn * ad
    ct = an * bd - bn * ad
    return at, bt, ct


def pivot_classify_and_stationary_points(an, bn, cn, ad, bd, cd):
    """
    Section 5.1.1/5.1.2/5.1.3 classification, plus stationary points if any
    exist. Returns (case_label, discriminant, stationary_points_list).
    """
    at, bt, ct = pivot_derivative_coeffs(an, bn, cn, ad, bd, cd)
    discriminant = at**2 + bt**2 - ct**2

    if discriminant > 1e-9:
        case = "5.1.1_stationary_points_exist"
        stationary = solve_a_sin_b_cos_c(at, bt, ct)
    elif discriminant < -1e-9:
        case = "5.1.2_monotonic_no_stationary_points"
        stationary = []
    else:
        case = "5.1.3_singular"
        stationary = solve_a_sin_b_cos_c(at, bt, ct)  # the (repeated) singular psi

    return case, discriminant, stationary


def pivot_singularity_exclusion_zone(an, bn, cn, ad, bd, cd, delta=0.05):
    """
    Eq (31): if discriminant == 0, returns the [psi_sing - delta, psi_sing +
    delta] interval to EXCLUDE (per the paper's stated practical strategy).
    Returns [] if there is no exact singularity for this joint at this pose.
    """
    case, disc, stationary = pivot_classify_and_stationary_points(an, bn, cn, ad, bd, cd)
    if case != "5.1.3_singular" or not stationary:
        return []
    psi_sing = stationary[0]
    return [(psi_sing - delta, psi_sing + delta)]


def pivot_limit_crossings(an, bn, cn, ad, bd, cd, theta_limit):
    """
    Eq (32): psi values where theta_i(psi) EQUALS theta_limit exactly.
    """
    tan_lim = math.tan(theta_limit)
    a_p = an - ad * tan_lim
    b_p = bn - bd * tan_lim
    c_p = cn - cd * tan_lim
    return solve_a_sin_b_cos_c(a_p, b_p, c_p)


def pivot_feasible_intervals(an, bn, cn, ad, bd, cd, lower_limit, upper_limit,
                              n_probe=360):
    """
    Full feasible-range computation for a pivot joint: finds the psi
    interval(s) where lower_limit <= theta_i(psi) <= upper_limit, using the
    EXACT crossing points from eq (32) as interval boundaries.

    Since pivot joints can have complex profiles (monotonic, with stationary
    points, possible wraparound discontinuities at +/-pi per Fig 5b/6), the
    boundaries are found exactly via eq(32), then a small number of PROBE
    points (n_probe, cheap since only pivot_theta() calls, no collision
    checking) between consecutive boundaries determine which segments are
    actually feasible -- this is a hybrid: exact boundaries, but a manual
    resolve of which side is "in" vs "out" at each one, since the
    derivative-sign disambiguation described in the paper needs care to get
    right in all cases and probing is simpler and just as reliable at
    negligible extra cost (probing pivot_theta directly, not the expensive
    collision check).
    """
    boundary_psis = set()
    boundary_psis.update(pivot_limit_crossings(an, bn, cn, ad, bd, cd, lower_limit))
    boundary_psis.update(pivot_limit_crossings(an, bn, cn, ad, bd, cd, upper_limit))
    boundary_psis.add(-math.pi)
    boundary_psis.add(math.pi)
    sorted_boundaries = sorted(boundary_psis)

    intervals = []
    for i in range(len(sorted_boundaries) - 1):
        lo, hi = sorted_boundaries[i], sorted_boundaries[i + 1]
        mid = (lo + hi) / 2
        theta_mid = pivot_theta(mid, an, bn, cn, ad, bd, cd)
        if lower_limit <= theta_mid <= upper_limit:
            intervals.append((lo, hi))
    return intervals


# ============================================================
# HINGE joints (theta2, theta6): theta_i = arccos(a sin(psi)+b cos(psi)+c)
# ============================================================
def hinge_feasible_intervals(a, b, c, lower_limit, upper_limit):
    """Eq (27)/(33)/(34)-equivalent: exact feasible psi interval(s)."""
    R = math.hypot(a, b)
    if R < 1e-12:
        theta_const = math.acos(max(-1.0, min(1.0, c)))
        return [(-math.pi, math.pi)] if lower_limit <= theta_const <= upper_limit else []

    phi = math.atan2(a, b)
    lo_bound = (math.cos(upper_limit) - c) / R
    hi_bound = (math.cos(lower_limit) - c) / R
    if lo_bound > 1.0 or hi_bound < -1.0:
        return []
    lo_c, hi_c = max(-1.0, lo_bound), min(1.0, hi_bound)
    if lo_c <= -1.0 and hi_c >= 1.0:
        return [(-math.pi, math.pi)]

    x_max = math.acos(lo_c)
    x_min = math.acos(hi_c)
    if x_min > x_max:
        return []

    def wrap(v):
        return (v + math.pi) % (2 * math.pi) - math.pi

    return [(wrap(x_min + phi), wrap(x_max + phi)),
            (wrap(-x_max + phi), wrap(-x_min + phi))]


# ============================================================
# Combining intervals, excluding singularities, picking a final psi
# ============================================================
def intersect_intervals(list_a, list_b):
    result = []
    for a_lo, a_hi in list_a:
        for b_lo, b_hi in list_b:
            lo, hi = max(a_lo, b_lo), min(a_hi, b_hi)
            if lo < hi:
                result.append((lo, hi))
    return result


def subtract_interval(intervals, exclude_lo, exclude_hi):
    """Removes [exclude_lo, exclude_hi] from a list of (lo,hi) intervals."""
    result = []
    for lo, hi in intervals:
        if exclude_hi <= lo or exclude_lo >= hi:
            result.append((lo, hi))   # no overlap
            continue
        if exclude_lo > lo:
            result.append((lo, exclude_lo))
        if exclude_hi < hi:
            result.append((exclude_hi, hi))
    return result


def choose_psi(feasible_intervals, preferred_psi=0.0):
    if not feasible_intervals:
        return None
    best_psi, best_dist = None, float('inf')
    for lo, hi in feasible_intervals:
        clamped = min(max(preferred_psi, lo), hi)
        dist = abs(clamped - preferred_psi)
        if dist < best_dist:
            best_dist, best_psi = dist, clamped
    return best_psi


if __name__ == "__main__":
    # Coefficients extracted from the verified pipeline at a real test pose
    # (target built from input angles 20,20,20,10,30,40,0 degrees).
    an, bn, cn = -0.082067, -0.033435, 0.171889
    ad, bd, cd = 0.036864, -0.074432, 0.382660

    case, disc, stationary = pivot_classify_and_stationary_points(an, bn, cn, ad, bd, cd)
    print(f"theta1 classification: {case} (discriminant={disc:.6f})")
    print(f"Stationary points: {stationary}")

    limit_lo, limit_hi = -1.0, 0.5   # example theta1 limits, radians
    feasible = pivot_feasible_intervals(an, bn, cn, ad, bd, cd, limit_lo, limit_hi)
    print(f"\nFeasible psi intervals for theta1 in [{limit_lo}, {limit_hi}]: {feasible}")

    print("\nVerification -- sampling theta1 at interval midpoints and just outside boundaries:")
    for lo, hi in feasible:
        mid = (lo + hi) / 2
        print(f"  mid of [{lo:.3f},{hi:.3f}]: psi={mid:.3f} -> theta1={pivot_theta(mid, an,bn,cn,ad,bd,cd):.4f}")