import numpy as np
from math import sqrt

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def print_solution(sol, dh, i, T):
    dh[:,0] = sol.T
    fk_solution = fk_compute(dh)
    print(f"Forward Kinematics for Solution {i+1}:")
    print(sol)
    print(fk_solution)
    print(T-fk_solution)


def theta1_sol(R_0_3, theta2, theta2_2):
    # theta1 = np.arctan2(-R_0_3[1,1], -R_0_3[0,1])
    # theta1_2 = np.pi + theta1
    # if theta1_2 > np.pi or theta1_2 < -np.pi:
    #     theta1_2 = wrap_to_pi(theta1_2)
    # return theta1, theta1_2

    ct1 = -R_0_3[0,1]/np.sin(theta2)
    st1 = -R_0_3[1,1]/np.sin(theta2)
    t1 = np.arctan2(st1, ct1)

    ct1 = -R_0_3[0,1]/np.sin(theta2_2)
    st1 = -R_0_3[1,1]/np.sin(theta2_2)
    t1_2 = np.arctan2(st1, ct1)

    return t1, t1_2



def theta2_sol(R_0_3, T, dh_table, l_s_e, l_e_w, l_s_w):
    ct2 = -R_0_3[2,1]
    st2 = sqrt(1 - ct2**2)
    theta2 = np.arctan2(st2, ct2)
    theta2_2 = np.arctan2(-st2, ct2)


    # endEffector_pos = np.array(T[0:3, 3], ndmin=2)
    # endEffector_pos = endEffector_pos.reshape((3,1))

    # # Extract orientation of end effector from T matrix
    # endEffector_o = np.array(T[0:3, 0:3], ndmin=2)
    # wrist_pos = endEffector_pos - dh_table[6,1] * (endEffector_o@np.array([[0], [0], [1]]))
    # alpha = np.arcsin((wrist_pos[2] - dh_table[0,1])/l_s_w).item()

    # beta = np.arccos((l_s_e**2 + l_s_w**2 - l_e_w**2)/(2*l_s_e*l_s_w)).item()
    # temp2 = alpha + beta # elbow up configuration
    # temp2_2 = alpha - beta # elbow down configuration


    # theta2 = np.pi/2 - (temp2)
    # # theta2_d = np.pi/2 - alpha - beta
    # theta2_2 = np.pi/2 - (temp2_2)

    # print(theta2, theta2_2)

    return theta2, theta2_2


def theta3_sol(R_0_3, theta2, theta2_2):
    # theta3 = np.arctan2(R_0_3[2,2], -R_0_3[2,0])
    # theta3_2 = theta3
    # return theta3, theta3_2
    ct3 = -R_0_3[2,0]/np.sin(theta2)
    st3 = R_0_3[2,2]/np.sin(theta2)
    t3 = np.arctan2(st3, ct3)

    ct3 = -R_0_3[2,0]/np.sin(theta2_2)
    st3 = R_0_3[2,2]/np.sin(theta2_2)
    t3_2 = np.arctan2(st3, ct3)

    return t3, t3_2




def theta4_sol(l_s_e, l_e_w, l_s_w):
    # These are in accordanc with dh axes
    ct4_d = (l_s_e**2 + l_e_w**2 - l_s_w**2)/(2*l_s_e*l_e_w)
    st4_d = sqrt(1 - ct4_d**2)
    t4_d_1 = np.arctan2(st4_d, ct4_d)
    t4_d_2 = np.arctan2(-st4_d, ct4_d)
    theta4 = np.pi - t4_d_1
    theta4_2 = np.pi - t4_d_2
    if theta4 > np.pi or theta4 < -np.pi:
        theta4 = wrap_to_pi(theta4)

    if theta4_2 > np.pi or theta4_2 < -np.pi:
        theta4_2 = wrap_to_pi(theta4_2)


    return theta4, theta4_2


def theta5_sol(R_4_7, theta6, theta6_2):
    ct5 = R_4_7[0,2]/np.sin(theta6)
    st5 = R_4_7[1,2]/np.sin(theta6)
    t5 = np.arctan2(st5,ct5)

    ct5 = R_4_7[0,2]/np.sin(theta6_2)
    st5 = R_4_7[1,2]/np.sin(theta6_2)
    t5_2 = np.arctan2(st5,ct5)
    return t5, t5_2



def theta6_sol(R_4_7):
    ct6 = R_4_7[2,2]
    st6 = sqrt(1 - ct6**2)
    t6 = np.arctan2(st6, ct6)
    t6_2 = np.arctan2(-st6, ct6)
    return t6, t6_2



def theta7_sol(R_4_7, theta6, theta6_2):
    ct7 = -R_4_7[2,0]/np.sin(theta6)
    st7 = R_4_7[2,1]/np.sin(theta6)
    t7 = np.arctan2(st7,ct7)

    ct7 = -R_4_7[2,0]/np.sin(theta6_2)
    st7 = R_4_7[2,1]/np.sin(theta6_2)
    t7_2 = np.arctan2(st7,ct7)
    return t7, t7_2


# will there be two reference planes based on theta1 and theta2? Yes
def calc_ref_plane_mat(dh_table, l_s_e, l_e_w, l_s_w, l_s_w_v, theta4, theta4_2, T):
    theta1_d = np.arctan2(l_s_w_v[1,0], l_s_w_v[0,0])
    print("Theta1 of reference plane: ", theta1_d)

    A = sqrt(l_s_w_v[1,0]**2 + l_s_w_v[0,0]**2)
    # A2 = -A # based on negative value of square root

    dew = dh_table[4,1]
    dse = dh_table[2,1]

    a = dew*np.sin(theta4)
    b = dew*np.cos(theta4) + dse
    z = l_s_w_v[2,0]

    theta2_d = np.arctan2((b*A - a*z), (a*A + b*z))

    a = dew*np.sin(theta4_2)
    b = dew*np.cos(theta4_2) + dse

    theta2_2_d = np.arctan2((b*A - a*z), (a*A + b*z))
    print("Theta2 of reference plane: ", theta2_d, theta2_2_d)
    

    # # ## this method is partially correct. Gives the values correct sometimes. Need to change from -np.pi/2 to np.pi/2 for negative values of theta2.
    # endEffector_pos = np.array(T[0:3, 3], ndmin=2)
    # endEffector_pos = endEffector_pos.reshape((3,1))

    # # Extract orientation of end effector from T matrix
    # endEffector_o = np.array(T[0:3, 0:3], ndmin=2)
    # wrist_pos = endEffector_pos - dh_table[6,1] * (endEffector_o@np.array([[0], [0], [1]]))
    # alpha = np.arcsin((wrist_pos[2] - dh_table[0,1])/l_s_w).item()

    # beta = np.arccos((l_s_e**2 + l_s_w**2 - l_e_w**2)/(2*l_s_e*l_s_w)).item()
    # temp2 = alpha + beta # elbow up configuration
    # temp2_2 = alpha - beta # elbow down configuration


    # theta2_d = -np.pi/2 + (temp2)
    # # theta2_d = np.pi/2 - alpha - beta
    # theta2_2_d = -np.pi/2 + (temp2_2)

    # print(theta2_d, theta2_2_d)

    # Obtained by substituting alpha and theta3 = 0 in homogenous transformation matrix of DH.
    R_0_1_p = np.array([[np.cos(theta1_d), 0, -np.sin(theta1_d)],
                        [np.sin(theta1_d), 0, np.cos(theta1_d)],
                        [0, -1, 0]])
    R_1_2_p = np.array([[np.cos(theta2_d), 0, np.sin(theta2_d)],
                        [np.sin(theta2_d), 0, -np.cos(theta2_d)],
                        [0, 1, 0]])
    R_2_3_p = np.array([[1, 0, 0],
                        [0, 0, 1],
                        [0, -1, 0]])
    
    ref_plane_mat = R_0_1_p@R_1_2_p@R_2_3_p

    R_1_2_p = np.array([[np.cos(theta2_2_d), 0, np.sin(theta2_2_d)],
                        [np.sin(theta2_2_d), 0, -np.cos(theta2_2_d)],
                        [0, 1, 0]])

    ref_plane_mat2 = R_0_1_p@R_1_2_p@R_2_3_p
    return ref_plane_mat, ref_plane_mat2

def calc_rot_0_4(theta1, theta2, theta3, theta4):
    R_0_1 = np.array([[np.cos(theta1), 0, -np.sin(theta1)],
                        [np.sin(theta1), 0, np.cos(theta1)],
                        [0, -1, 0]])
    R_1_2 = np.array([[np.cos(theta2), 0, np.sin(theta2)],
                        [np.sin(theta2), 0, -np.cos(theta2)],
                        [0, 1, 0]])
    R_2_3 = np.array([[np.cos(theta3), 0,  -np.sin(theta3)],
                    [np.sin(theta3), 0,   np.cos(theta3)],
                    [0,              -1,  0            ]])

    R_3_4 = np.array([[np.cos(theta4), 0, np.sin(theta4)],
                    [np.sin(theta4), 0, -np.cos(theta4)],
                    [0,              1,  0]])
    
    return R_0_1@R_1_2@R_2_3@R_3_4


def calc_rot_4_7(T, R_0_4):
    R_0_7 = np.array(T[0:3, 0:3], ndmin=2)

    return R_0_4.T@R_0_7

def calc_psi_plane_mat(R_psi, ref_plane_mat):
# premultiplying as R_psi is wrt fixed base link
# R_0_3_p keeps takes us to 3 in reference plane. While we know the R_psi which gives the plane according to selected elbow angle psi. 
# So to take to actual plane we can premultiply by R_psi as R_psi is with respect to base link fixed frame.
# Other way is to know the R_2_3 i.e with respect to 2 the angle psi. Then we can post multiply.
    return R_psi@ref_plane_mat


def dh_transform(a, alpha, d, theta):

    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,       sa,       ca,      d],
        [0,        0,        0,      1]
    ])


def fk_compute(dh):

    # Initialize the transformation matrix as an identity matrix
    T = np.eye(4)
    # for d, a, alpha, theta in dh:
    for theta, d, a, alpha in dh:
        T = T @ dh_transform(a, alpha, d, theta)

    return T

def sw_vector_compute(dh, T):

    shoulder_pos = np.array([[0],[0],[dh[0,1]]])
    wrist_tool = np.array([[0],[0],[dh[6,1]]])

    # Extract position of end effector from T matrix
    endEffector_pos = np.array(T[0:3, 3], ndmin=2)
    endEffector_pos = endEffector_pos.reshape((3,1))

    # Extract orientation of end effector from T matrix
    endEffector_o = np.array(T[0:3, 0:3], ndmin=2)

    x_sw = endEffector_pos - shoulder_pos - endEffector_o@wrist_tool
    return x_sw


def calculate_orientation(t1234, T):

    R_0_4 = calc_rot_0_4(t1234[0], t1234[1], t1234[2], t1234[3])

    R_4_7 = calc_rot_4_7(T, R_0_4)

    theta6_1_rad, theta6_2_rad = theta6_sol(R_4_7)
    theta5_1_rad, theta5_2_rad = theta5_sol(R_4_7, theta6_1_rad, theta6_2_rad)
    theta7_1_rad, theta7_2_rad = theta7_sol(R_4_7, theta6_1_rad, theta6_2_rad)

    solution_1 = np.zeros(7)
    solution_2 = np.zeros(7)

    solution_1[0:4] = t1234[0:4]
    solution_2[0:4] = t1234[0:4]

    solution_1[4] = theta5_1_rad
    solution_1[5] = theta6_1_rad
    solution_1[6] = theta7_1_rad

    solution_2[4] = theta5_2_rad
    solution_2[5] = theta6_2_rad
    solution_2[6] = theta7_2_rad

    return solution_1, solution_2

def arm_angle_rot_mat():

    pass

