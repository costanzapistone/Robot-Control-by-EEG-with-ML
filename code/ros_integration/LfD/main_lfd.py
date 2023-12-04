# %%
from Learning_from_demonstration import LfD
import rospy
#%%
LfD=LfD()
LfD.home_gripper() # homeing the gripper allows to kinestheicall move it. 
rospy.sleep(5)


#%%
LfD.home()
#%%
LfD.traj_rec()
#%%
LfD.save(file="left")
#%%
LfD.load(file="left")
#%%
LfD.home()
#%%
LfD.traj_rec()
#%%
LfD.save(file="right")
# Execute left trajectory
#%%
LfD.load(file="right")
#%%
LfD.home_gripper() # homeing the gripper allows to kinestheicall move it. 
rospy.sleep(2)
LfD.execute()
# Execute left trajectory
#%%
LfD.load(file="right")
#%%
LfD.home_gripper() # homeing the gripper allows to kinestheicall move it. 
rospy.sleep(2)
LfD.execute()
# %%
LfD.home()
# %%
