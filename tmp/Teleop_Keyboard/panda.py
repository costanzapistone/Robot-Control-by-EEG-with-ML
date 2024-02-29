#%%
#!/usr/bin/env python
import rospy
import math
import numpy as np
import quaternion # pip install numpy-quaternion
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, WrenchStamped, Twist
from std_msgs.msg import Float32MultiArray
import dynamic_reconfigure.client
from pose_transform_functions import  array_quat_2_pose
from franka_gripper.msg import GraspActionGoal, HomingActionGoal, StopActionGoal, MoveActionGoal

class Panda():
    """
    Base class to interact with Franka Emika robot, through the 
    nodes and topics provided by franka ROS and the 
    franka_human_friendly_controller package
    """
    def __init__(self):
        super(Panda, self).__init__()

        # variables to store the cartesian pose of the robot end effector 
        self.curr_pos = None
        self.curr_ori = None
        self.gripper_width = None
                
        # Set up ROS subscribers and publishers
        
        self.pos_sub = rospy.Subscriber("/cartesian_pose", PoseStamped, self.ee_pos_callback)
        self.force_feedback_sub = rospy.Subscriber('/force_torque_ext', WrenchStamped, self.force_feedback_callback)
        self.goal_pub = rospy.Publisher('/equilibrium_pose', PoseStamped, queue_size=0)
        self.configuration_pub = rospy.Publisher('/equilibrium_configuration', Float32MultiArray, queue_size=0)
        self.grasp_pub = rospy.Publisher("/franka_gripper/grasp/goal", GraspActionGoal, queue_size=0)
        self.move_pub = rospy.Publisher("/franka_gripper/move/goal", MoveActionGoal, queue_size=0)
        self.homing_pub = rospy.Publisher("/franka_gripper/homing/goal", HomingActionGoal, queue_size=0)
        self.stop_pub = rospy.Publisher("/franka_gripper/stop/goal", StopActionGoal, queue_size=0)
    
        # Ask 

        self.force_feedback = 0.0
        # self.set_K = dynamic_reconfigure.client.Client('/dynamic_reconfigure_compliance_param_node', config_callback = None)
        self.joint_states_sub = rospy.Subscriber("/joint_states", JointState, self.joint_states_callback)
       
        self.grasp_goal = GraspActionGoal()
        self.homing_goal = HomingActionGoal()
        self.stop_goal = StopActionGoal()
        self.move_goal = MoveActionGoal()
        
        self.move_goal.goal.speed = 1
        self.grasp_goal.goal.epsilon.inner = 0.3
        self.grasp_goal.goal.epsilon.outer = 0.3
        self.grasp_goal.goal.speed = 0.1
        self.grasp_goal.goal.force = 5
        self.grasp_goal.goal.width = 0.03

        rospy.sleep(2)

    # Callback function to update the current end-effector position and orientation

    def ee_pos_callback(self, curr_conf):
        """
        Callback function that updates the cartesian position of the Robots 
        end effector, as it changes, from the '/cartesian_pose' toplllic provided
        by franka_ros
        """
        # Set the current position 

        self.curr_pos = np.array([curr_conf.pose.position.x, curr_conf.pose.position.y, curr_conf.pose.position.z])
        # Set the current orientation 

        self.curr_ori = np.array([curr_conf.pose.orientation.w, curr_conf.pose.orientation.x, curr_conf.pose.orientation.y, curr_conf.pose.orientation.z])

    def force_feedback_callback(self, feedback):
        self.force = feedback.wrench.force
        self.force_feedback = np.linalg.norm(np.array([self.force.x, self.force.y, self.force.z]))


    def grasp_gripper(self, width, force):
        self.grasp_goal.goal.width = width  # Larghezza dell'apertura del gripper
        self.grasp_goal.goal.force = force  # Forza di presa
        self.grasp_pub.publish(self.grasp_goal)

    def homing_gripper(self):
        self.homing_pub.publish(self.homing_goal)

    def stop_gripper(self):
        self.stop_pub.publish(self.stop_goal)

    def move_gripper(self, width):
        self.move_goal.goal.width = width  # Larghezza dell'apertura del gripper
        self.move_goal.goal.speed = 0.1
        #self.move_goal.header.seq = 1
        self.move_pub.publish(self.move_goal)

    def joint_states_callback(self, data):
        self.curr_joint = data.position[:7]
        self.gripper_width = data.position[7] + data.position[8]
        
    def set_stiffness(self, k_t1, k_t2, k_t3,k_r1,k_r2,k_r3, k_ns):
        
        self.set_K.update_configuration({"translational_stiffness_X": k_t1})
        self.set_K.update_configuration({"translational_stiffness_Y": k_t2})
        self.set_K.update_configuration({"translational_stiffness_Z": k_t3})
        self.set_K.update_configuration({"rotational_stiffness_X": k_r1}) 
        self.set_K.update_configuration({"rotational_stiffness_Y": k_r2}) 
        self.set_K.update_configuration({"rotational_stiffness_Z": k_r3})
        self.set_K.update_configuration({"nullspace_stiffness": k_ns}) 
