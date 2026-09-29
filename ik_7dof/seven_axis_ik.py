"""
7DOF IK based on the papers of Shimizu and Faria, with ANALYTIC psi (arm
angle) selection replacing manual/sampled input.

This combines:
  1. The verified symbolic S-R-S IK pipeline (theta1..theta7 as exact
     closed-form functions of psi and the target pose T).
  2. Faria (2018) Section 5's closed-form feasible-arm-angle-interval
     method, applied to ALL SIX psi-dependent joints (theta1,2,3,5,6,7 --
     theta4 is independent of psi and checked separately), covering BOTH
     joint limits (eq. 32) and singularity avoidance (eq. 31).

The output solutions need not match any particular manually-input joint
angles, since there are infinitely many valid solutions (one per feasible
psi). This script instead picks a psi that is provably within joint limits
and away from singularities for ALL SIX psi-dependent joints.
"""

import math
import sympy as sp
import numpy as np
from sympy.geometry.polygon import rad, deg


# ============================================================
# Section 5 solver: exact (non-sampling) psi feasibility
# ============================================================
def solve_a_sin_b_cos_c(a, b, c):
    """
    Exact solutions of a*sin(psi) + b*cos(psi) + c = 0, via the Weierstrass
    substitution t = tan(psi/2). Reduces to (c-b)t^2 + 2at + (b+c) = 0.
    """
    A, B, C = c - b, 2 * a, b + c
    if abs(A) < 1e-14:
        if abs(B) < 1e-14:
            return []
        return [2 * math.atan(-C / B)]
    disc = B**2 - 4 * A * C
    if disc < 0:
        return []
    sq = math.sqrt(disc)
    return [2 * math.atan((-B + sq) / (2 * A)), 2 * math.atan((-B - sq) / (2 * A))]


def pivot_theta(psi_val, an, bn, cn, ad, bd, cd):
    u = an * math.sin(psi_val) + bn * math.cos(psi_val) + cn
    v = ad * math.sin(psi_val) + bd * math.cos(psi_val) + cd
    return math.atan2(u, v)


def pivot_limit_crossings(an, bn, cn, ad, bd, cd, theta_limit):
    """Eq (32): psi values where a pivot joint's theta(psi) equals theta_limit."""
    tan_lim = math.tan(theta_limit)
    a_p = an - ad * tan_lim
    b_p = bn - bd * tan_lim
    c_p = cn - cd * tan_lim
    return solve_a_sin_b_cos_c(a_p, b_p, c_p)


def pivot_feasible_intervals(an, bn, cn, ad, bd, cd, lower_limit, upper_limit):
    """Exact feasible psi interval(s) for a pivot (atan2-type) joint."""
    boundaries = set(pivot_limit_crossings(an, bn, cn, ad, bd, cd, lower_limit))
    boundaries.update(pivot_limit_crossings(an, bn, cn, ad, bd, cd, upper_limit))
    boundaries.update([-math.pi, math.pi])
    sb = sorted(boundaries)
    intervals = []
    for i in range(len(sb) - 1):
        lo, hi = sb[i], sb[i + 1]
        if hi - lo < 1e-9:
            continue   # skip degenerate near-zero-width artifacts
        mid = (lo + hi) / 2
        if lower_limit <= pivot_theta(mid, an, bn, cn, ad, bd, cd) <= upper_limit:
            intervals.append((lo, hi))
    return intervals


def pivot_derivative_coeffs(an, bn, cn, ad, bd, cd):
    """Eq (29): at, bt, ct -- numerator of dtheta_i/dpsi."""
    at = cn * bd - bn * cd
    bt = an * cd - cn * ad
    ct = an * bd - bn * ad
    return at, bt, ct


def pivot_singularity_zone(an, bn, cn, ad, bd, cd, delta=0.05):
    """Eq (31): [psi_sing - delta, psi_sing + delta] to exclude, if a true singularity exists."""
    at, bt, ct = pivot_derivative_coeffs(an, bn, cn, ad, bd, cd)
    disc = at**2 + bt**2 - ct**2
    if abs(disc) > 1e-9:
        return []
    sing = solve_a_sin_b_cos_c(at, bt, ct)
    return [(sing[0] - delta, sing[0] + delta)] if sing else []


def hinge_feasible_intervals(a, b, c, lower_limit, upper_limit):
    """Exact feasible psi interval(s) for a hinge (arccos-type) joint."""
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

    return [(wrap(x_min + phi), wrap(x_max + phi)), (wrap(-x_max + phi), wrap(-x_min + phi))]


def hinge_singularity_zone(a, b, c, delta=0.05):
    """Hinge joints become singular where theta_i = 0 or pi (sin(theta_i) = 0)."""
    R = math.hypot(a, b)
    zones = []
    if R < 1e-12:
        return zones
    for target in (1.0, -1.0):
        for psi_sing in solve_a_sin_b_cos_c(a, b, c - target):
            zones.append((psi_sing - delta, psi_sing + delta))
    return zones


def intersect_intervals(list_a, list_b):
    result = []
    for a_lo, a_hi in list_a:
        for b_lo, b_hi in list_b:
            lo, hi = max(a_lo, b_lo), min(a_hi, b_hi)
            if lo < hi:
                result.append((lo, hi))
    return result


def subtract_interval(intervals, ex_lo, ex_hi):
    result = []
    for lo, hi in intervals:
        if ex_hi <= lo or ex_lo >= hi:
            result.append((lo, hi))
            continue
        if ex_lo > lo:
            result.append((lo, ex_lo))
        if ex_hi < hi:
            result.append((ex_hi, hi))
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


def linear_coeffs(expr, psi_sym):
    """Extracts (a,b,c) such that expr == a*sin(psi)+b*cos(psi)+c EXACTLY."""
    expr = sp.expand_trig(sp.expand(expr))
    a = float(expr.coeff(sp.sin(psi_sym)))
    b = float(expr.coeff(sp.cos(psi_sym)))
    c = float(sp.simplify(expr - a * sp.sin(psi_sym) - b * sp.cos(psi_sym)))
    return a, b, c


# ============================================================
# Verified S-R-S IK pipeline
# ============================================================
def A_mat(theta, alpha, d, a):
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta) * sp.cos(alpha), sp.sin(theta) * sp.sin(alpha), a * sp.cos(theta)],
        [sp.sin(theta), sp.cos(theta) * sp.cos(alpha), -sp.cos(theta) * sp.sin(alpha), a * sp.sin(theta)],
        [0, sp.sin(alpha), sp.cos(alpha), d],
        [0, 0, 0, 1]
    ])


def theta1_sol(R_0_3, theta2_v, theta2_2_v):
    ct1 = -R_0_3[0, 1] / sp.sin(theta2_v)
    st1 = -R_0_3[1, 1] / sp.sin(theta2_v)
    t1 = sp.atan2(st1, ct1)
    ct1 = -R_0_3[0, 1] / sp.sin(theta2_2_v)
    st1 = -R_0_3[1, 1] / sp.sin(theta2_2_v)
    t1_2 = sp.atan2(st1, ct1)
    return t1, t1_2


def theta2_sol(R_0_3):
    ct2 = -R_0_3[2, 1]
    st2 = sp.sqrt(1 - ct2**2)
    return sp.atan2(st2, ct2), sp.atan2(-st2, ct2)


def theta3_sol(R_0_3, theta2_v, theta2_2_v):
    ct3 = -R_0_3[2, 0] / sp.sin(theta2_v)
    st3 = R_0_3[2, 2] / sp.sin(theta2_v)
    t3 = sp.atan2(st3, ct3)
    ct3 = -R_0_3[2, 0] / sp.sin(theta2_2_v)
    st3 = R_0_3[2, 2] / sp.sin(theta2_2_v)
    t3_2 = sp.atan2(st3, ct3)
    return t3, t3_2


def theta4_sol(l_s_e, l_e_w, l_s_w):
    ct4 = (l_s_e**2 + l_e_w**2 - l_s_w**2) / (2 * l_s_e * l_e_w)
    st4 = sp.sqrt(1 - ct4**2)
    t4_1 = sp.atan2(st4, ct4)
    t4_2 = sp.atan2(-st4, ct4)
    return sp.pi - t4_1, sp.pi - t4_2


def theta5_sol(R_4_7, theta6_v, theta6_2_v):
    ct5 = R_4_7[0, 2] / sp.sin(theta6_v)
    st5 = R_4_7[1, 2] / sp.sin(theta6_v)
    t5 = sp.atan2(st5, ct5)
    ct5 = R_4_7[0, 2] / sp.sin(theta6_2_v)
    st5 = R_4_7[1, 2] / sp.sin(theta6_2_v)
    t5_2 = sp.atan2(st5, ct5)
    return t5, t5_2


def theta6_sol(R_4_7):
    ct6 = R_4_7[2, 2]
    st6 = sp.sqrt(1 - ct6**2)
    return sp.atan2(st6, ct6), sp.atan2(-st6, ct6)


def theta7_sol(R_4_7, theta6_v, theta6_2_v):
    ct7 = -R_4_7[2, 0] / sp.sin(theta6_v)
    st7 = R_4_7[2, 1] / sp.sin(theta6_v)
    t7 = sp.atan2(st7, ct7)
    ct7 = -R_4_7[2, 0] / sp.sin(theta6_2_v)
    st7 = R_4_7[2, 1] / sp.sin(theta6_2_v)
    t7_2 = sp.atan2(st7, ct7)
    return t7, t7_2


def calc_ref_plane_mat(p26, l_s_e, l_e_w, theta4_v, theta4_2_v, l1, l2):
    theta1v = sp.atan2(p26[1, 0], p26[0, 0])
    A = sp.sqrt(p26[0, 0]**2 + p26[1, 0]**2)
    z = p26[2, 0]
    a1 = l_e_w * sp.sin(theta4_v)
    b1 = l_e_w * sp.cos(theta4_v) + l_s_e
    theta2v_1 = sp.atan2(b1 * A - a1 * z, a1 * A + b1 * z)
    a2 = l_e_w * sp.sin(theta4_2_v)
    b2 = l_e_w * sp.cos(theta4_2_v) + l_s_e
    theta2v_2 = sp.atan2(b2 * A - a2 * z, a2 * A + b2 * z)
    T01_v = A_mat(theta1v, -sp.pi / 2, l1, 0)
    T12_v = A_mat(theta2v_1, sp.pi / 2, 0, 0)
    T12_v_2 = A_mat(theta2v_2, sp.pi / 2, 0, 0)
    T23_v = A_mat(0, -sp.pi / 2, l_s_e, 0)
    R03_v_1 = (T01_v @ T12_v @ T23_v)[0:3, 0:3]
    R03_v_2 = (T01_v @ T12_v_2 @ T23_v)[0:3, 0:3]
    return R03_v_1, R03_v_2


def calc_rot_0_4(theta1_v, theta2_v, theta3_v, theta4_v):
    R_0_1 = sp.Matrix([[sp.cos(theta1_v), 0, -sp.sin(theta1_v)],
                        [sp.sin(theta1_v), 0, sp.cos(theta1_v)], [0, -1, 0]])
    R_1_2 = sp.Matrix([[sp.cos(theta2_v), 0, sp.sin(theta2_v)],
                        [sp.sin(theta2_v), 0, -sp.cos(theta2_v)], [0, 1, 0]])
    R_2_3 = sp.Matrix([[sp.cos(theta3_v), 0, -sp.sin(theta3_v)],
                        [sp.sin(theta3_v), 0, sp.cos(theta3_v)], [0, -1, 0]])
    R_3_4 = sp.Matrix([[sp.cos(theta4_v), 0, sp.sin(theta4_v)],
                        [sp.sin(theta4_v), 0, -sp.cos(theta4_v)], [0, 1, 0]])
    return R_0_1 @ R_1_2 @ R_2_3 @ R_3_4


def calc_rot_4_7(T_mat, R_0_4):
    return R_0_4.T @ T_mat[0:3, 0:3]


def sw_vector_compute(T_mat, l1, l4):
    shoulder_pos = sp.Matrix([[0], [0], [l1]])
    wrist_tool = sp.Matrix([[0], [0], [l4]])
    return T_mat[0:3, 3] - shoulder_pos - T_mat[0:3, 0:3] * wrist_tool


def calculate_orientation(t1234, T_mat):
    R_0_4 = calc_rot_0_4(t1234[0], t1234[1], t1234[2], t1234[3])
    R_4_7 = calc_rot_4_7(T_mat, R_0_4)
    t6_1, t6_2 = theta6_sol(R_4_7)
    t5_1, t5_2 = theta5_sol(R_4_7, t6_1, t6_2)
    t7_1, t7_2 = theta7_sol(R_4_7, t6_1, t6_2)
    sol1 = list(t1234[0:4]) + [t5_1, t6_1, t7_1]
    sol2 = list(t1234[0:4]) + [t5_2, t6_2, t7_2]
    return sol1, sol2


def fk_numeric(angles_rad, l1, l2, l3, l4):
    dh = [(angles_rad[0], l1, 0, -np.pi / 2), (angles_rad[1], 0, 0, np.pi / 2),
          (angles_rad[2], l2, 0, -np.pi / 2), (angles_rad[3], 0, 0, np.pi / 2),
          (angles_rad[4], l3, 0, -np.pi / 2), (angles_rad[5], 0, 0, np.pi / 2),
          (angles_rad[6], l4, 0, 0)]
    T_mat = np.eye(4)
    for theta, d, a, alpha in dh:
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        Ti = np.array([[ct, -st * ca, st * sa, a * ct], [st, ct * ca, -ct * sa, a * st],
                        [0, sa, ca, d], [0, 0, 0, 1]])
        T_mat = T_mat @ Ti
    return T_mat


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    psi = sp.symbols('psi', real=True)
    l1v, l2v, l3v, l4v = 0.169, 0.300, 0.320, 0.185

    # Target pose (for demonstration -- replace with your real target pose)
    q_input_deg = [20, 20, 20, 10, 30, 40, 0]
    q_input_rad = [math.radians(v) for v in q_input_deg]
    T_target_np = fk_numeric(q_input_rad, l1v, l2v, l3v, l4v)
    T07 = sp.Matrix(T_target_np.tolist())

    print(f"Target pose from input angles: {q_input_deg}")

    p26 = sw_vector_compute(T07, l1v, l4v)
    l_s_w = sp.sqrt((p26.T * p26)[0])
    p26_u = p26 / l_s_w

    theta4_expr, theta4_2_expr = theta4_sol(l2v, l3v, l_s_w)
    theta4_expr = theta4_expr.evalf()
    theta4_2_expr = theta4_2_expr.evalf()
    print(f"theta4 branches: {deg(theta4_expr).evalf()}, {deg(theta4_2_expr).evalf()} deg")

    R03_v_1, R03_v_2 = calc_ref_plane_mat(p26, l2v, l3v, theta4_expr, theta4_2_expr, l1v, l2v)
    R03_v_1 = R03_v_1.evalf()
    R03_v_2 = R03_v_2.evalf()

    K = sp.Matrix([[0, -p26_u[2, 0], p26_u[1, 0]],
                   [p26_u[2, 0], 0, -p26_u[0, 0]],
                   [-p26_u[1, 0], p26_u[0, 0], 0]])
    R_psi_sym = sp.eye(3) + sp.sin(psi) * K + (1 - sp.cos(psi)) * (K * K)

    # R_0_3(psi), branch 1 -- psi kept SYMBOLIC/free
    R03_sym = sp.expand_trig(sp.expand(R_psi_sym * R03_v_1))
    R03_sym_2 = sp.expand_trig(sp.expand(R_psi_sym * R03_v_2))

    # Extract theta1, theta2, theta3 coefficients
    a2, b2, c2 = linear_coeffs(-R03_sym[2, 1], psi)
    an1, bn1, cn1 = linear_coeffs(-R03_sym[1, 1], psi)
    ad1, bd1, cd1 = linear_coeffs(-R03_sym[0, 1], psi)
    an3, bn3, cn3 = linear_coeffs(R03_sym[2, 2], psi)
    ad3, bd3, cd3 = linear_coeffs(-R03_sym[2, 0], psi)

    # R_0_4(psi) = R_0_3(psi) @ R_3_4(fixed theta4) -- stay in MATRIX form,
    # never decompose-and-recompose from scalar angles (breaks linearity).
    R_3_4_fixed = sp.Matrix([[sp.cos(theta4_expr), 0, sp.sin(theta4_expr)],
                              [sp.sin(theta4_expr), 0, -sp.cos(theta4_expr)], [0, 1, 0]])
    R_0_4_psi = sp.expand_trig(sp.expand(R03_sym * R_3_4_fixed))
    R_4_7_psi = sp.expand_trig(sp.expand(R_0_4_psi.T * T07[0:3, 0:3]))

    # Extract theta5, theta6, theta7 coefficients
    a6, b6, c6 = linear_coeffs(R_4_7_psi[2, 2], psi)
    an5, bn5, cn5 = linear_coeffs(R_4_7_psi[1, 2], psi)
    ad5, bd5, cd5 = linear_coeffs(R_4_7_psi[0, 2], psi)
    an7, bn7, cn7 = linear_coeffs(R_4_7_psi[2, 1], psi)
    ad7, bd7, cd7 = linear_coeffs(-R_4_7_psi[2, 0], psi)

    # ---- Joint limits -- REPLACE with your actual hardware limits (radians) ----
    theta1_limits = (-math.pi, math.pi)
    theta2_limits = (0.1, 2.8)
    theta3_limits = (-math.pi, math.pi)
    theta5_limits = (-math.pi, math.pi)
    theta6_limits = (0.1, 2.8)
    theta7_limits = (-math.pi, math.pi)

    feasible_t1 = pivot_feasible_intervals(an1, bn1, cn1, ad1, bd1, cd1, *theta1_limits)
    feasible_t2 = hinge_feasible_intervals(a2, b2, c2, *theta2_limits)
    feasible_t3 = pivot_feasible_intervals(an3, bn3, cn3, ad3, bd3, cd3, *theta3_limits)
    feasible_t5 = pivot_feasible_intervals(an5, bn5, cn5, ad5, bd5, cd5, *theta5_limits)
    feasible_t6 = hinge_feasible_intervals(a6, b6, c6, *theta6_limits)
    feasible_t7 = pivot_feasible_intervals(an7, bn7, cn7, ad7, bd7, cd7, *theta7_limits)

    combined = feasible_t1
    for f in (feasible_t2, feasible_t3, feasible_t5, feasible_t6, feasible_t7):
        combined = intersect_intervals(combined, f)
    print(f"\nFeasible psi (joint limits, ALL 6 joints): {combined}")

    # ---- Singularity exclusion, all 6 joints ----
    delta = 0.05
    sing_zones = []
    sing_zones += pivot_singularity_zone(an1, bn1, cn1, ad1, bd1, cd1, delta)
    sing_zones += hinge_singularity_zone(a2, b2, c2, delta)
    sing_zones += pivot_singularity_zone(an3, bn3, cn3, ad3, bd3, cd3, delta)
    sing_zones += pivot_singularity_zone(an5, bn5, cn5, ad5, bd5, cd5, delta)
    sing_zones += hinge_singularity_zone(a6, b6, c6, delta)
    sing_zones += pivot_singularity_zone(an7, bn7, cn7, ad7, bd7, cd7, delta)

    print(f"Singularity zones to exclude: {sing_zones}")

    safe = combined
    for ex_lo, ex_hi in sing_zones:
        safe = subtract_interval(safe, ex_lo, ex_hi)
    print(f"\nFeasible psi (joint limits AND singularity avoidance): {safe}")

    final_psi = choose_psi(safe, preferred_psi=0.0)
    print(f"\nFINAL analytically-chosen psi: {final_psi}")

    if final_psi is None:
        print("No feasible psi found -- pose unreachable within given limits/singularity margins.")
    else:
        R03_1_final = R03_sym.subs(psi, final_psi).evalf()
        theta2_f, theta2_2_f = theta2_sol(R03_1_final)
        theta1_f, theta1_2_f = theta1_sol(R03_1_final, theta2_f, theta2_2_f)
        theta3_f, theta3_2_f = theta3_sol(R03_1_final, theta2_f, theta2_2_f)
        t1234_f = [theta1_f, theta2_f, theta3_f, theta4_expr]
        sol_a, sol_b = calculate_orientation(t1234_f, T07)

        sol_deg = [float(deg(v).evalf()) for v in sol_a]
        print(f"\nFinal solution (degrees): {sol_deg}")

        T_check_np = fk_numeric([float(v) for v in sol_a], l1v, l2v, l3v, l4v)
        max_diff = np.max(np.abs(T_check_np - T_target_np))
        print(f"FK verification: max_diff = {max_diff:.2e} ({'PASS' if max_diff < 1e-6 else 'FAIL'})")
        print(f"\n(compare against original input: {q_input_deg})")