
#%%

import os
os.chdir("/home/andrei/Desktop/PROJECT_ELLIS_COMDO/FOLDER_code")
import comdo.utils as utils


#%%
utils.test1()

alphas = [3]     # going alpha < 3 solves the shimmering, but still doesn't converge to optimum

betas = [0.2]
utils.plot_streamlined(alphas, betas, n_iterations=100, step_version= "step_sequential")

# %%
