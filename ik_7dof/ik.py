import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
from ik_7dof import ik_fn as ik

class ik_publisher_7dof(Node):

    def __init__(self):
        super().__init__('ik_7dof')

        # self.angles = np.array([0.54, 2.88, 1.475, 0.5, 0.54, 0.51, -1.22])
        self.angles = np.array([-2.072, 0.841, 2.004, 0.613, 3.159, -0.878, 0.577])                
        self.sols = self.calculate_ik(self.angles)

        self.i = 0
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)
        timer_period = 2  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        

    def calculate_ik(self, angles_rad):

        self.dh_table = np.array([[0, 0.2221, 0, -np.pi/2],
                            [0, 0,      0,  np.pi/2],
                            [0, 0.3,    0, -np.pi/2],
                            [0, 0,      0,  np.pi/2],
                            [0, 0.320,   0, -np.pi/2],
                            [0, 0,      0,  np.pi/2],
                            [0, 0.225,  0,  0      ]])
        
        self.dh_table[:,0] = angles_rad.T.flatten()
        self.T = ik.fk_compute(self.dh_table)

        # arm angle
        self.psi = np.pi/2

        self.l_s_e = self.dh_table[2,1] # Length from shoulder to elbow
        self.l_e_w = self.dh_table[4,1] # length from elbow to wrist

        self.shoulder_pos = np.array([[0],[0],[self.dh_table[0,1]]])

        self.l_s_w_v = ik.sw_vector_compute(self.dh_table, self.T)

        self.l_s_w_u = self.l_s_w_v/np.linalg.norm(self.l_s_w_v) # Unit vector from shoulder to wrist
        self.l_s_w = np.linalg.norm(self.l_s_w_v) # Length from shoulder to wrist

        self.k_s_w = np.array([[0, -self.l_s_w_u[2,0], self.l_s_w_u[1,0]],
                               [self.l_s_w_u[2,0],  0, -self.l_s_w_u[0,0]],
                               [-self.l_s_w_u[1,0], self.l_s_w_u[0,0], 0]])
        
        #Rodrigues Formula
        self.R_psi = np.eye(3) + (1 - np.cos(self.psi))*(self.k_s_w@self.k_s_w) + np.sin(self.psi)*self.k_s_w

        self.theta4, self.theta4_2 = ik.theta4_sol(self.l_s_e, self.l_e_w, self.l_s_w)

        self.R_ref, self.R_ref2 = ik.calc_ref_plane_mat(self.dh_table, self.l_s_e, self.l_e_w, self.l_s_w, self.l_s_w_v, \
                                                        self.theta4_2, self.theta4, self.T)


        self.R_0_3 = ik.calc_psi_plane_mat(self.R_psi, self.R_ref)
        self.R_0_3_2 = ik.calc_psi_plane_mat(self.R_psi, self.R_ref2)

        self.theta2, self.theta2_2 = ik.theta2_sol(self.R_0_3, self.T, self.dh_table, self.l_s_e, self.l_e_w, self.l_s_w)
        self.theta2_3, self.theta2_4 = ik.theta2_sol(self.R_0_3_2, self.T, self.dh_table, self.l_s_e, self.l_e_w, self.l_s_w)
        self.theta1, self.theta1_2 = ik.theta1_sol(self.R_0_3, self.theta2, self.theta2_2)
        self.theta1_3, self.theta1_4 = ik.theta1_sol(self.R_0_3_2, self.theta2_3, self.theta2_4) # Gives the same values as theta1 and theta1_2
        self.theta3, self.theta3_2 = ik.theta3_sol(self.R_0_3, self.theta2, self.theta2_2)
        self.theta3_3, self.theta3_4 = ik.theta3_sol(self.R_0_3_2, self.theta2_3, self.theta2_4)

        self.t1234_1 = np.array([self.theta1, self.theta2, self.theta3, self.theta4_2])
        self.t1234_2 = np.array([self.theta1_3, self.theta2_3, self.theta3_3, self.theta4]) # elbow down config for same theta1
        self.t1234_3 = np.array([self.theta1_2, self.theta2_2, self.theta3, self.theta4])
        self.t1234_4 = np.array([self.theta1_4, self.theta2_4, self.theta3_3, self.theta4_2]) #elbow down config for same theta1

        self.sol1, self.sol2 = ik.calculate_orientation(self.t1234_1, self.T)
        self.sol3, self.sol4 = ik.calculate_orientation(self.t1234_2, self.T)
        self.sol5, self.sol6 = ik.calculate_orientation(self.t1234_3, self.T)
        self.sol7, self.sol8 = ik.calculate_orientation(self.t1234_4, self.T)

        sol = np.array([self.sol1, self.sol2, self.sol3, self.sol4, self.sol5, self.sol6, self.sol7, self.sol8])

        for i in range(len(sol)):
            ik.print_solution(sol[i], self.dh_table, i, self.T)

        return sol      


    def timer_callback(self):
        if (self.i > 7):
            self.i = 0

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['J1_', 'J2_', 'J3_', 'J4_', 'J5_', 'J6_', 'J7_' ]
        msg.position = self.sols[self.i].tolist()
        msg.velocity = []
        msg.effort = []

        self.publisher_.publish(msg)

        # self.get_logger().info(f"Joint positions: {msg.position}")
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    publisher_ = ik_publisher_7dof()

    rclpy.spin(publisher_)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    publisher_.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


