'''
7DOF IK based on the papers of Shimizu and Faria. 
Inputs to this are joint angles and arm angle.

The output solutions need not match the exact angles of the input as here we have infinite solutions
depending on the arm angle. Atleast one solution of the output angle will match our input if 
our arm angle matches the implicit arm angle of the input. 
'''

import sympy as sp
import numpy as np
from sympy.geometry.polygon import rad, deg


dh_table = np.array([[0, 0.2221, 0, -np.pi/2],
                    [0, 0,      0,  np.pi/2],
                    [0, 0.3,    0, -np.pi/2],
                    [0, 0,      0,  np.pi/2],
                    [0, 0.320,   0, -np.pi/2],
                    [0, 0,      0,  np.pi/2],
                    [0, 0.225,  0,  0      ]])



theta1, d1, a1, alpha1 = sp.symbols('theta_1 d_1 a_1 alpha_1')
theta2, d2, a2, alpha2 = sp.symbols('theta_2 d_2 a_2 alpha_2')
theta3, d3, a3, alpha3 = sp.symbols('theta_3 d_3 a_3 alpha_3')
theta4, d4, a4, alpha4 = sp.symbols('theta_4 d_4 a_4 alpha_4')
theta5, d5, a5, alpha5 = sp.symbols('theta_5 d_5 a_5 alpha_5')
theta6, d6, a6, alpha6 = sp.symbols('theta_6 d_6 a_6 alpha_6')
theta7, d7, a7, alpha7 = sp.symbols('theta_7 d_7 a_7 alpha_7')
psi                    = sp.symbols('psi')
l1, l2, l3, l4         = sp.symbols('l_1 l_2 l_3 l_4')
t11, t12, t13, t14, t21, t22, t23, t24, t31, t32, t33, t34 = sp.symbols('t_11, t_12, t_13, t_14, t_21, t_22, t_23, t_24, t_31, t_32, t_33, t_34')
theta1v, theta2v, theta3v, theta4v = sp.symbols('θ₁ᵛ, θ₂ᵛ, θ₃ᵛ, θ₄ᵛ')

T = sp.Matrix([[t11, t12, t13, t14],
               [t21, t22, t23, t24],
               [t31, t32, t33, t34],
               [0, 0, 0, 1]])

p02 = sp.Matrix([0, 0, l1]).T
p24 = sp.Matrix([0, l2, 0]).T
p46 = sp.Matrix([0, 0, l3]).T
p67 = sp.Matrix([0, 0, l4]).T


def wrap_to_pi(angle):
    return (angle + sp.pi) % (2 * sp.pi) - sp.pi


def theta1_sol(R_0_3, theta2, theta2_2):
    ct1 = -R_0_3[0,1]/sp.sin(theta2)
    st1 = -R_0_3[1,1]/sp.sin(theta2)
    t1 = sp.atan2(st1, ct1)

    ct1 = -R_0_3[0,1]/sp.sin(theta2_2)
    st1 = -R_0_3[1,1]/sp.sin(theta2_2)
    t1_2 = sp.atan2(st1, ct1)

    return t1, t1_2


def theta2_sol(R_0_3):
    ct2 = -R_0_3[2,1]
    st2 = sp.sqrt(1 - ct2**2)
    theta2 = sp.atan2(st2, ct2)
    theta2_2 = sp.atan2(-st2, ct2)

    return theta2, theta2_2


def theta3_sol(R_0_3, theta2, theta2_2):
    ct3 = -R_0_3[2,0]/sp.sin(theta2)
    st3 = R_0_3[2,2]/sp.sin(theta2)
    t3 = sp.atan2(st3, ct3)

    ct3 = -R_0_3[2,0]/sp.sin(theta2_2)
    st3 = R_0_3[2,2]/sp.sin(theta2_2)
    t3_2 = sp.atan2(st3, ct3)

    return t3, t3_2


def theta4_sol(l_s_e, l_e_w, l_s_w):
    ct4_d = (l_s_e**2 + l_e_w**2 - l_s_w**2)/(2*l_s_e*l_e_w)
    st4_d = sp.sqrt(1 - ct4_d**2)
    t4_d_1 = sp.atan2(st4_d, ct4_d)
    t4_d_2 = sp.atan2(-st4_d, ct4_d)
    theta4 = sp.pi - t4_d_1
    theta4_2 = sp.pi - t4_d_2
    if theta4 > sp.pi or theta4 < -sp.pi:
        theta4 = wrap_to_pi(theta4)

    if theta4_2 > sp.pi or theta4_2 < -sp.pi:
        theta4_2 = wrap_to_pi(theta4_2)


    return theta4, theta4_2


def theta5_sol(R_4_7, theta6, theta6_2):
    ct5 = R_4_7[0,2]/sp.sin(theta6)
    st5 = R_4_7[1,2]/sp.sin(theta6)
    t5 = sp.atan2(st5,ct5)

    ct5 = R_4_7[0,2]/sp.sin(theta6_2)
    st5 = R_4_7[1,2]/sp.sin(theta6_2)
    t5_2 = sp.atan2(st5,ct5)
    return t5, t5_2


def theta6_sol(R_4_7):
    ct6 = R_4_7[2,2]
    st6 = sp.sqrt(1 - ct6**2)
    t6 = sp.atan2(st6, ct6)
    t6_2 = sp.atan2(-st6, ct6)
    return t6, t6_2



def theta7_sol(R_4_7, theta6, theta6_2):
    ct7 = -R_4_7[2,0]/sp.sin(theta6)
    st7 = R_4_7[2,1]/sp.sin(theta6)
    t7 = sp.atan2(st7,ct7)

    ct7 = -R_4_7[2,0]/sp.sin(theta6_2)
    st7 = R_4_7[2,1]/sp.sin(theta6_2)
    t7_2 = sp.atan2(st7,ct7)
    return t7, t7_2

'''
Does 2 reference plane solutions for each theta1v theta2v makes sense?
'''
def calc_ref_plane_mat(p26, l_s_e, l_e_w, theta4, theta4_2):

    theta1v = sp.atan2(p26[1, 0], p26[0, 0])
 
    A = sp.sqrt(p26[0, 0]**2 + p26[1, 0]**2)
    z = p26[2, 0]
 
    a1 = l_e_w * sp.sin(theta4)
    b1 = l_e_w * sp.cos(theta4) + l_s_e
    theta2v_1 = sp.atan2(b1 * A - a1 * z, a1 * A + b1 * z)
 
    a2 = l_e_w * sp.sin(theta4_2)
    b2 = l_e_w * sp.cos(theta4_2) + l_s_e
    theta2v_2 = sp.atan2(b2 * A - a2 * z, a2 * A + b2 * z)


    T01_v = A_mat(theta1v, alpha1, l1, a1)
    T01_v = T01_v.subs({alpha1:-sp.pi/2, a1:0})

    T12_v = A_mat(theta2v_1, sp.pi/2, 0, 0)
    T12_v_2 = A_mat(theta2v_2, sp.pi/2, 0, 0)
    T23_v = A_mat(0, -sp.pi/2, l2, 0)

    R03_v_1 = (T01_v@T12_v@T23_v)[0:3, 0:3]
    R03_v_2 = (T01_v@T12_v_2@T23_v)[0:3, 0:3]
    # sp.pprint(R03_v)

    # R_1_2_p = sp.Matrix([[sp.cos(theta2_2_d), 0, sp.sin(theta2_2_d)],
    #                     [sp.sin(theta2_2_d), 0, -sp.cos(theta2_2_d)],
    #                     [0, 1, 0]])

    return R03_v_1, R03_v_2



def calc_rot_0_4(theta1, theta2, theta3, theta4):
    R_0_1 = sp.Matrix([[sp.cos(theta1), 0, -sp.sin(theta1)],
                        [sp.sin(theta1), 0, sp.cos(theta1)],
                        [0, -1, 0]])
    R_1_2 = sp.Matrix([[sp.cos(theta2), 0, sp.sin(theta2)],
                        [sp.sin(theta2), 0, -sp.cos(theta2)],
                        [0, 1, 0]])
    R_2_3 = sp.Matrix([[sp.cos(theta3), 0,  -sp.sin(theta3)],
                    [sp.sin(theta3), 0,   sp.cos(theta3)],
                    [0,              -1,  0            ]])

    R_3_4 = sp.Matrix([[sp.cos(theta4), 0, sp.sin(theta4)],
                    [sp.sin(theta4), 0, -sp.cos(theta4)],
                    [0,              1,  0]])
    
    return R_0_1@R_1_2@R_2_3@R_3_4


def calc_rot_4_7(T, R_0_4):
    R_0_7 = sp.Matrix(T[0:3, 0:3], ndmin=2)

    return R_0_4.T@R_0_7

def calc_psi_plane_mat(R_psi, ref_plane_mat):
# premultiplying as R_psi is wrt fixed base link
# R_0_3_p keeps takes us to 3 in reference plane. While we know the R_psi which gives the plane according to selected elbow angle psi. 
# So to take to actual plane we can premultiply by R_psi as R_psi is with respect to base link fixed frame.
# Other way is to know the R_2_3 i.e with respect to 2 the angle psi. Then we can post multiply.
    return R_psi@ref_plane_mat


def dh_transform(a, alpha, d, theta):

    ct = sp.cos(theta)
    st = sp.sin(theta)
    ca = sp.cos(alpha)
    sa = sp.sin(alpha)

    return sp.Matrix([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,       sa,       ca,      d],
        [0,        0,        0,      1]
    ])


def fk_compute(dh):

    # Initialize the transformation matrix as an identity matrix
    T = sp.eye(4)
    # for d, a, alpha, theta in dh:
    for theta, d, a, alpha in dh:
        T = T @ dh_transform(a, alpha, d, theta)

    return T

def sw_vector_compute(T):

    shoulder_pos = sp.Matrix([[0],[0],[l1]])
    wrist_tool = sp.Matrix([[0],[0],[l4]])

    # Extract position of end effector from T matrix
    endEffector_pos = sp.Matrix(T[0:3, 3], ndmin=2)
    # endEffector_pos = endEffector_pos.reshape((3,1))

    # Extract orientation of end effector from T matrix
    endEffector_o = sp.Matrix(T[0:3, 0:3], ndmin=2)

    x_sw = endEffector_pos - shoulder_pos - endEffector_o@wrist_tool
    return x_sw


def calculate_orientation(t1234, T):

    R_0_4 = calc_rot_0_4(t1234[0], t1234[1], t1234[2], t1234[3])

    R_4_7 = calc_rot_4_7(T, R_0_4)

    theta6_1_rad, theta6_2_rad = theta6_sol(R_4_7)
    theta5_1_rad, theta5_2_rad = theta5_sol(R_4_7, theta6_1_rad, theta6_2_rad)
    theta7_1_rad, theta7_2_rad = theta7_sol(R_4_7, theta6_1_rad, theta6_2_rad)

    solution_1 = [0,0,0,0,0,0,0]
    solution_2 = [0,0,0,0,0,0,0]

    solution_1[0:4] = t1234[0:4]
    solution_2[0:4] = t1234[0:4]

    solution_1[4] = theta5_1_rad
    solution_1[5] = theta6_1_rad
    solution_1[6] = theta7_1_rad

    solution_2[4] = theta5_2_rad
    solution_2[5] = theta6_2_rad
    solution_2[6] = theta7_2_rad

    return solution_1, solution_2

# Define the DH transformation matrix function
def A_mat(theta, alpha, d, a):
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha),  sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
        [sp.sin(theta),  sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
        [0,              sp.sin(alpha),               sp.cos(alpha),                d],
        [0,              0,                           0,                            1]
    ])



# Build transformation matrices
A01 = A_mat(theta1, alpha1, d1, a1)
A12 = A_mat(theta2, alpha2, d2, a2)
A23 = A_mat(theta3, alpha3, d3, a3)
A34 = A_mat(theta4, alpha4, d4, a4)
A45 = A_mat(theta5, alpha5, d5, a5)
A56 = A_mat(theta6, alpha6, d6, a6)
A67 = A_mat(theta7, alpha7, d7, a7)


# Compute forward kinematics up to frame 4
T07 = A01 * A12 * A23 * A34 * A45 * A56 * A67


T07 = T07.subs({
    a1: 0, a2:0, a3: 0, a4: 0, a5:0, a6:0, a7:0,
    d1: l1, d2:0, d3: l2, d4: 0, d5:l3, d6:0, d7:l4,
    alpha1: -sp.pi/2,
    alpha2: sp.pi/2,
    alpha3: -sp.pi/2,
    alpha4: sp.pi/2,
    alpha5: -sp.pi/2,
    alpha6: sp.pi/2,
    alpha7: 0
})

T07 = T07.subs({l1:0.169, l2:0.300, l3:0.320, l4:0.185})

T07_t = T07.subs({l1:0.169, l2:0.300, l3:0.320, l4:0.185})

j1 = 20
j2 = 20
j3 = 20
j4 = 10
j5 = 30
j6 = 40
j7 = 0
arm_angle = rad(90)

sp.pprint("----------------------")
sp.pprint(f"Input Joint Angles: {j1}, {j2}, {j3}, {j4}, {j5}, {j6}, {j7}")
sp.pprint("----------------------")

T07 = T07.subs({theta1:rad(j1),
                theta2:rad(j2),
                theta3:rad(j3),
                theta4:rad(j4),
                theta5:rad(j5),
                theta6:rad(j6),
                theta7:rad(j7)})

T07 = T07.evalf()

sp.pprint("Transformation matrix for the input joint angles")
sp.pprint(T07)
sp.pprint("----------------------")


# # Shoulder to wrist data
p26 = sw_vector_compute(T07)
p26 = p26.subs({l1:0.169, l2:0.300, l3:0.320, l4:0.185})
p26_u = p26/p26.norm().evalf() # unit vector
p26_distance = p26.norm().evalf()

sp.pprint(f"Shoulder Wrist distance: {p26_distance}")
sp.pprint("----------------------")

## Solving for joint 4 as it is independent of the arm angle
theta4_expr, theta4_2_expr = theta4_sol(0.300, 0.320, p26_distance)
theta4_expr = theta4_expr.subs({l1:0.169, l2:0.300, l3:0.320, l4:0.185}).evalf()
theta4_2_expr = theta4_2_expr.subs({l1:0.169, l2:0.300, l3:0.320, l4:0.185}).evalf() 
sp.pprint(f"Joint 4 solutions: {deg(theta4_expr).evalf()}, {deg(theta4_2_expr).evalf()}")
sp.pprint("----------------------")


#### Skew symmetric matrix of shoulder to wrist vector. This is used in Rodrigues rotation formula
# sp.pprint("Skew Symmetric matrix of shoulder wrist vector")
p26_cross = sp.Matrix([[0, -p26_u[2,0], p26_u[1,0]],
                        [p26_u[2,0],  0, -p26_u[0,0]],
                        [-p26_u[1,0], p26_u[0,0], 0]])

# R_psi represents rotation of angle psi about the shoulder wrist vector in world coordinates
sp.pprint("arm angle matrix with respect to world:")
R_psi = (sp.eye(3) + (1 - sp.cos(psi))*(p26_cross@p26_cross) + sp.sin(psi)*p26_cross).subs({psi:arm_angle}).evalf()
sp.pprint(R_psi)
sp.pprint("----------------------")


# Calculate reference or virtual plane. We assume theta3 = 0 for the virtual plane. theta4 can be independent of the arm angle.
sp.pprint("Reference plane rotation matrices: ")
R03_v_1, R03_v_2 = calc_ref_plane_mat(p26, 0.300, 0.320, theta4_expr, theta4_2_expr)
R03_v_1 = R03_v_1.evalf()
R03_v_2 = R03_v_2.evalf()

sp.pprint(R03_v_1)
sp.pprint(R03_v_2)
sp.pprint("----------------------")

sp.pprint("Actual Arm  plane rotation matrices: ")
# # We get two solutions corresponding to two different reference plane solutions
R03_1 = calc_psi_plane_mat(R_psi, R03_v_1).evalf() # theta4 has no role to play for the plane
R03_2 = calc_psi_plane_mat(R_psi, R03_v_2).evalf()

sp.pprint(R03_1) ##branch 1
sp.pprint(R03_2) ##branch 2

sp.pprint("=" * 60)
sp.pprint("COMPUTING ALL 8 SOLUTIONS")
sp.pprint("=" * 60)

# ---- Branch 1: R03_1 paired with theta4_expr ----
theta2_b1, theta2_2_b1 = theta2_sol(R03_1)
theta1_b1, theta1_2_b1 = theta1_sol(R03_1, theta2_b1, theta2_2_b1)
theta3_b1, theta3_2_b1 = theta3_sol(R03_1, theta2_b1, theta2_2_b1)

# ---- Branch 2: R03_2 paired with theta4_2_expr ----
theta2_b2, theta2_2_b2 = theta2_sol(R03_2)
theta1_b2, theta1_2_b2 = theta1_sol(R03_2, theta2_b2, theta2_2_b2)
theta3_b2, theta3_2_b2 = theta3_sol(R03_2, theta2_b2, theta2_2_b2)


# ---- Assemble the 4 (theta1,theta2,theta3,theta4) shoulder/elbow configs ----
# Each pairs theta1/theta2/theta3 that were derived TOGETHER (same sign choice)
t1234_list = [
    [theta1_b1,   theta2_b1,   theta3_b1,   theta4_expr],    # branch 1a
    [theta1_2_b1, theta2_2_b1, theta3_2_b1, theta4_expr],    # branch 1b
    [theta1_b2,   theta2_b2,   theta3_b2,   theta4_2_expr],  # branch 2a
    [theta1_2_b2, theta2_2_b2, theta3_2_b2, theta4_2_expr],  # branch 2b
]

# ---- Expand each into 2 wrist solutions via calculate_orientation -> 8 total ----
all_solutions = []
for idx, t1234 in enumerate(t1234_list):
    sol_a, sol_b = calculate_orientation(t1234, T07)
    all_solutions.append(sol_a)
    all_solutions.append(sol_b)

sp.pprint(f"Total solutions assembled: {len(all_solutions)}")
sp.pprint("=" * 60)

# ---- Print each solution in degrees, and verify via FK round-trip ----
for i, sol in enumerate(all_solutions):
    sol_deg = [deg(sol[j]).evalf() for j in range(7)]
    sp.pprint(f"Solution {i+1}: {sol_deg}")

    T_check = T07_t.subs({
        theta1: sol[0], theta2: sol[1], theta3: sol[2], theta4: sol[3],
        theta5: sol[4], theta6: sol[5], theta7: sol[6]
    }).evalf()

    diff = (T_check - T07).evalf()
    max_diff = max(abs(diff[r, c]) for r in range(4) for c in range(4))
    status = "PASS" if max_diff < 1e-6 else f"FAIL (max diff = {max_diff})"
    sp.pprint(f"  FK verification: {status}")
    sp.pprint("-" * 40)

sp.pprint("(compare against input: 20, 20, 20, 10, 30, 40, 0)")