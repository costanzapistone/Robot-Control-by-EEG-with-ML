#%%
from Learning_from_demonstration import LfD
import rospy
from scipy.io import loadmat
from EEGClass import EEGClass

#%%
# Load the .mat file you want to evaluate
mat_file = '/home/costanza/Robot-Control-by-EEG-with-ML/data/BCICIV_calib_ds1a.mat'
data = loadmat(mat_file, struct_as_record=True)

#%%
# Create an istance of the class EEGClass
eeg_instance = EEGClass(data)

# Segment the data
trials = eeg_instance.segmentation()

# Compute the FFT
fft_trials = eeg_instance.fft(trials)

# Compute the LDA for dimensionality reduction
X_train, X_test, y_train, y_test = eeg_instance.lda(fft_trials)


#%% 
# Load the trained model
import joblib

trained_model = joblib.load('/home/costanza/Robot-Control-by-EEG-with-ML/trained_model/trained_model_best.joblib')

#%%
# Predict the labels of the test set
y_pred = trained_model.predict(X_test)
print(y_pred.shape)

# %%
def execute_labeled_trajectory(label):
    # Load and execute the corresponding trajectory based on the label
    if label == -1:
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
for label in y_pred:
    execute_labeled_trajectory(label)
    LfD.home()
    rospy.sleep(5)
    print("Trajectory executed")
    # Wait for the trajectory to finish and the robot to return home
    #while not LfD.is_at_home():
    #    rospy.sleep(1)

LfD.home()