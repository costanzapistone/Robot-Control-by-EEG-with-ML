#%%
from Learning_from_demonstration import LfD
import rospy
import csv

#%%

def execute_labeled_trajectory(label):
    # Load and execute the corresponding trajectory based on the label
    if label == 0:
        trajectory_file = "left"
    else:
        trajectory_file = "right"

    # Load and execute the trajectory
    LfD.load(trajectory_file)
    LfD.home_gripper() # homeing the gripper allows to kinestheicall move it.
    LfD.execute()
    rospy.sleep(5)

#%%

LfD=LfD()
LfD.home_gripper() # homeing the gripper allows to kinestheicall move it. 
rospy.sleep(5)

LfD.home()

#%%

# Read labels from CSV file
with open('/home/costanza/Robot-Control-by-EEG-with-ML/code/predicted_labels/predicted_labels.csv', 'r') as csvfile:
    label_reader = csv.reader(csvfile)
    # Predicted labels are in the first row
    labels = [float(label) for label in next(label_reader)]

    for label in labels:
        execute_labeled_trajectory(label)
        LfD.home()
        rospy.sleep(5)
        print("Trajectory executed")
        # Wait for the trajectory to finish and the robot to return home
        #while not LfD.is_at_home():
        #    rospy.sleep(1)

# Assuming you have a method to go back home
LfD.go_to_home()

# Save the recorded trajectory if needed
#LfD.save()


# %%
