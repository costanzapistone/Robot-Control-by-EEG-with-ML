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
LfD.home()
#%%
#LfD.home_gripper() # homeing the gripper allows to kinestheicall move it. 
#%%
LfD.load(file="left")
#%%
LfD.execute()
#%%
LfD.home()
#%%
LfD.traj_rec()
#%%
LfD.save(file="right")
#%%
LfD.home()
#%%
#LfD.home_gripper() # homeing the gripper allows to kinestheicall move it. 
#%%
LfD.load(file="right")
#%%
LfD.execute()
# %%
