import rospy
import math
import numpy as np
from geometry_msgs.msg import PoseStamped, Pose, Twist
from pynput.keyboard import Listener
from pose_transform_functions import  array_quat_2_pose, list_2_quaternion, position_2_array, transform_pose
from panda import Panda
from scipy.io import loadmat
from EEGClass import EEGClass
from processing_functions import get_random_segment, lda_one_segment
import joblib

class TeleopKeyboard(Panda):
    """
    A class that allows a user to control the cartesian pose of the Franka Emika robots through 
    a key board, that builds on top of the funcionality provided by the `Panda` class,
    which provides an interface to the Franka Emika robots. This is done through the `teleop_twist_keyboard`
    package.
    """
    # Define constants
    SUBJECT = 'a'
    MAT_FILE = f'/home/costanza/Robot-Control-by-EEG-with-ML/data/BCICIV_calib_ds1{SUBJECT}.mat'
    TRAINED_MODEL_LR = '/home/costanza/Robot-Control-by-EEG-with-ML/trained_model/trained_model_best.joblib'
    
    # Trained models for the Ensamble Learning
    TRAINED_MODEL_DT = '/home/costanza/Robot-Control-by-EEG-with-ML/trained_model/trained_model_DT.joblib'
    TRAINED_MODEL_KNN = '/home/costanza/Robot-Control-by-EEG-with-ML/trained_model/trained_model_KNN.joblib'
    TRAINED_MODEL_SVM = '/home/costanza/Robot-Control-by-EEG-with-ML/trained_model/trained_model_SVM.joblib'
    TRAINED_MODEL_NB = '/home/costanza/Robot-Control-by-EEG-with-ML/trained_model/trained_model_NB.joblib'

    def __init__(self):
        # start up a ROS node
        rospy.init_node("teleop_node")
        super(TeleopKeyboard, self).__init__()

        # Set the node frequency - twice the frequency of the /key_vel topic
        self.r = rospy.Rate(20)
        
        # Variable to store the current key value.
        self.key_value = None

        # Variable to store the MI class
        self.key = None

        # List to store the key sequences
        self.label_sequence = [] 
        
        # Subscriber to keyboard from teleop_twist_keyboard package
        self.key_sub = rospy.Subscriber("/cmd_vel", Twist, self.keyboard_read_callback)
        
        # Amount to move the robot 
        self.move_distance = 0.005 

        # This funciton is needed to keep the node running 
        rospy.spin()

    
    def move_up(self):
        """
        Move the robot up by move_distance
        """
        # Set this to the current value as it needs some orientation value to publish to /equilibrium_pose
        quat_goal = list_2_quaternion(self.curr_ori)
        # Get current postion and add 1 mm to the z axis
        goal = array_quat_2_pose(self.curr_pos + np.array([0.0, 0.0, self.move_distance]), quat_goal)
        self.goal_pub.publish(goal)
        self.key_value.linear.x = 0.0
        self.key_value.angular.z = 0.0
        print("moved Robot up 5 mm")

    def move_down(self):
        """
        Move the robot down by move_distance
        """
        # Set this to the current value as it needs some orientation value to publish to /equilibrium_pose
        quat_goal = list_2_quaternion(self.curr_ori)
        # Get current postion and add 1 mm to the z axis
        goal = array_quat_2_pose(self.curr_pos + np.array([0.0, 0.0, -self.move_distance]), quat_goal)
        self.goal_pub.publish(goal)
        self.key_value.linear.x = 0.0
        self.key_value.angular.z = 0.0
        print("moved Robot down 5 mm")

    def move_left(self):
        """
        Move the robot left by move_distance
        """
        # Set this to the current value as it needs some orientation value to publish to /equilibrium_pose
        quat_goal = list_2_quaternion(self.curr_ori)
        # Get current postion and add 1 mm to the z axis
        goal = array_quat_2_pose(self.curr_pos + np.array([0.0, self.move_distance, 0.0]), quat_goal)
        self.goal_pub.publish(goal)
        self.key_value.linear.x = 0.0
        self.key_value.angular.z = 0.0
        print("moved Robot left 5 mm")

    def move_right(self):
        """
        Move the robot right by move_distance
        """
        # Set this to the current value as it needs some orientation value to publish to /equilibrium_pose
        quat_goal = list_2_quaternion(self.curr_ori)
        # Get current postion and add 1 mm to the z axis
        goal = array_quat_2_pose(self.curr_pos + np.array([0.0, -self.move_distance, 0.0]), quat_goal)
        self.goal_pub.publish(goal)
        self.key_value.linear.x = 0.0
        self.key_value.angular.z = 0.0
        print("moved Robot right 5 mm")

    def move_forward(self):
        """
        Move the robot forward by move_distance
        """
        # Set this to the current value as it needs some orientation value to publish to /equilibrium_pose
        quat_goal = list_2_quaternion(self.curr_ori)
        # Get current postion and add 1 mm to the z axis
        goal = array_quat_2_pose(self.curr_pos + np.array([self.move_distance, 0.0, 0.0]), quat_goal)
        self.goal_pub.publish(goal)
        self.key_value.linear.x = 0.0
        self.key_value.angular.z = 0.0
        print("moved Robot forward 5 mm")

    def move_backward(self):
        """
        Move the robot backward by move_distance
        """
        # Set this to the current value as it needs some orientation value to publish to /equilibrium_pose
        quat_goal = list_2_quaternion(self.curr_ori)
        # Get current postion and add 5 mm to the z axis
        goal = array_quat_2_pose(self.curr_pos + np.array([-self.move_distance, 0.0, 0.0]), quat_goal)
        self.goal_pub.publish(goal)
        self.key_value.linear.x = 0.0
        self.key_value.angular.z = 0.0
        print("moved Robot backward 5 mm")
    
    def open_gripper(self):
        """
        Open the gripper by move_distance
        """
        gripper_width = self.gripper_width + self.move_distance
        self.move_gripper(gripper_width)
        self.key_value.linear.x = 0.0
        self.key_value.angular.z = 0.0
        print("Opened the gripper of 5 mm")

    def close_gripper(self):
        """
        Close the gripper by move_distance
        """
        gripper_width = self.gripper_width - self.move_distance
        self.move_gripper(gripper_width)
        self.key_value.linear.x = 0.0
        self.key_value.angular.z = 0.0
        print("Closed the gripper of 5 mm")    
        
    def majority_vote(self, predictions):
        """
        Perform majority voting on a list of multiple binary predictions from the same trained model.

        Parameters:
        - predictions (list): List of binary predictions (-1 or +1).

        Returns:
        - int: Majority voted prediction (+1 or -1).
        """
        if not predictions:
            raise ValueError("Empty list of predictions.")

        # Count occurrences of each class in the list
        count_minus1 = predictions.count(-1)
        count_plus1 = predictions.count(1)

        # Determine the majority class
        if count_minus1 > count_plus1:
            return -1
        elif count_plus1 > count_minus1:
            return 1
        else:
            # If there's a tie, you can handle it based on your specific needs
            # For simplicity, this implementation returns -1 in case of a tie
            return -1
        
    def keyboard_read_callback(self, key_input):
        """
        Callback function that changes the robots end effector cartesian pose when 
        one of the arrow keys are pressed. It changes the equilibrium pose by 5mm, given the key pressed
        and direction assigned to that key.
        
        In the method below, the key pressed is used to predict the movement of the robot using the trained model.

        Key - Action         
        'i' = + z-axis
        ',' = - z-axis
        'j' = + y-axis
        'l' = - y-axis
        'u' = + x-axis
        'o' = - x-axis
        'm' = open the gripper
        '.' = close the gripper
        """
        self.key_value = key_input
        print(self.key_value)

        # # Define the subject
        # subject = 'b'
        # # Load the .mat file
        # mat_file = f'/home/costanza/Robot-Control-by-EEG-with-ML/data/BCICIV_calib_ds1{subject}.mat'
        data = loadmat(self.MAT_FILE, struct_as_record=True)
        # Create an instance of the class EEGClass
        eeg_instance = EEGClass(data)
        # Segment the data
        trials = eeg_instance.segmentation()
        # Compute the FFT
        fft_trials = eeg_instance.fft(trials)

        # Based on the key pressed, pick a random segment from a different key of the dictionary fft_trials.
        if self.key_value.linear.x == 0.0 and self.key_value.angular.z == 1.0:
            # Get key 'left'
            self.key = list(fft_trials.keys())[0]
            
        elif self.key_value.linear.x == 0.0 and self.key_value.angular.z == -1.0:
            # Get key 'right'
            self.key = list(fft_trials.keys())[1]
        
        ############################# Ensamble Learning - Majority Voting #########################################
        # Get a random segment based on the key
        segment = get_random_segment(fft_trials, self.key)

        # Apply LDA
        X_test_lda = lda_one_segment(segment, self.SUBJECT)

        # Load the trained models
        trained_model_lr = joblib.load(self.TRAINED_MODEL_LR)
        trained_model_dt = joblib.load(self.TRAINED_MODEL_DT)
        trained_model_knn = joblib.load(self.TRAINED_MODEL_KNN)
        trained_model_svm = joblib.load(self.TRAINED_MODEL_SVM)
        trained_model_nb = joblib.load(self.TRAINED_MODEL_NB)

        # Create a list of the trained models
        trained_models = [trained_model_lr, trained_model_dt, trained_model_knn, trained_model_svm, trained_model_nb]

        # Use the trained models to predict the class of the segment
        predictions_list = []
        for model in trained_models:
            y_pred = model.predict(X_test_lda)
            predictions_list.append(y_pred)
        print("Predicted movement from the ML models :", predictions_list)
        # Perform majority voting on the list of predictions
        y_pred = self.majority_vote(predictions_list)

        print("Predicted movement from Majority Voting :", y_pred)

        # Get 10 random segments
        # random_segments = get_random_segments(fft_trials, self.key, num_segments=10)

        # # Create a list to store the predictions
        # predictions_list = []

        # # Iterate the same process over the k segments
        # for segment in random_segments:
        #     # Apply LDA
        #     X_test_lda = lda_one_segment(segment, self.SUBJECT)

        #     # Load the trained model
        #     trained_model = joblib.load(self.TRAINED_ML_MODEL_PATH)

        #     # Use the trained model to predict the class of the segment
        #     y_pred = trained_model.predict(X_test_lda)

        #     # Append the prediction to the list
        #     predictions_list.append(y_pred)

        # # Perform majority voting on the list of predictions
        # y_pred = self.majority_vote(predictions_list)

        # # Print the predicted label
        # print("Predicted movement from the ML model from Majority Voting :", y_pred)



        # # Get a random segment based on the key
        # segment = get_random_segment(fft_trials, self.key)

        # # Apply LDA
        # X_test_lda = lda_one_segment(segment, self.SUBJECT)

        # # Load the trained model
        # trained_model = joblib.load(self.TRAINED_ML_MODEL_PATH)

        # # Use the trained model to predict the class of the segment
        # y_pred = trained_model.predict(X_test_lda)

        # # Print the predicted label
        # print("Predicted movement from the ML model :", y_pred)

        # # If the predicted label is -1, move the robot to the left
        # if y_pred == -1:
        #     self.key_value.linear.x = 0.0
        #     self.key_value.angular.z = 1.0
        # # If the predicted label is 1, move the robot to the right
        # elif y_pred == 1:
        #     self.key_value.linear.x = 0.0
        #     self.key_value.angular.z = -1.0

        ############################# Sequence of 2 labels - Performing 2^2 actions #########################################

        # Append the label to the list
        self.label_sequence.append(y_pred)

        # Check if the list has 2 elements
        if len(self.label_sequence) == 2:
            print("Predicted Sequence:", self.label_sequence)
            # Determine the action to perform based on the sequence of 2 labels
            if self.label_sequence == [1, 1]:
                # Move the robot right
                self.key_value.linear.x = 0.0
                self.key_value.angular.z = -1.0
                self.move_right()
            
            elif self.label_sequence == [-1, 1]:
                # Move the robot up
                self.key_value.linear.x = 0.5
                self.key_value.angular.z = 0.0
                self.move_up()

            elif self.label_sequence == [1, -1]:
                # Move the robot down
                self.key_value.linear.x = -0.5
                self.key_value.angular.z = 0.0
                self.move_down()

            elif self.label_sequence == [-1, -1]:
                # Move the robot left
                self.key_value.linear.x = 0.0
                self.key_value.angular.z = 1.0
                self.move_left()

            # Reset the list
            self.label_sequence = []

        else:
            # If the list has less than 2 elements, do not move the robot
            self.key_value.linear.x = 0.0
            self.key_value.angular.z = 0.0

                            

        # # Move along + Z axis
        # if self.key_value.linear.x == 0.5 and self.key_value.angular.z == 0.0:
        #     self.move_up()

        # # - Z axis
        # elif self.key_value.linear.x == -0.5 and self.key_value.angular.z == 0.0:
        #     self.move_down()

        # # + Y axis - j key - left
        # elif self.key_value.linear.x == 0.0 and self.key_value.angular.z == 1.0:
        #     self.move_left()
         
        # # - Y axis - l key - right
        # elif self.key_value.linear.x == 0.0 and self.key_value.angular.z == -1.0:
        #     self.move_right()

        # # + X axis
        # elif self.key_value.linear.x == 0.5 and self.key_value.angular.z == 1.0:
        #     self.move_forward()
       
        # # - X axis
        # elif self.key_value.linear.x == 0.0 and self.key_value.angular.z == -1.0:
        #     self.move_backward()

        # # Open the gripper (press m)
        # elif self.key_value.linear.x == -0.5 and self.key_value.angular.z == -1.0:
        #     self.open_gripper()     

        # # Close the gripper (press .)
        # elif self.key_value.linear.x == -0.5 and self.key_value.angular.z == 1.0:
        #     self.close_gripper()
