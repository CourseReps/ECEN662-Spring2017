import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pandas as pd
from itertools import compress

# get raw data for real images
dfr = pd.read_csv('real.csv')

# remove "outliers"
dfr_array = np.array(dfr).flatten()
dfr_mask = np.array(dfr_array < 0.04)
mask_list = dfr_mask.ravel().tolist()
dfr_array = list(compress(dfr_array,mask_list))
dfr_inliers = pd.DataFrame(dfr_array)
rmean = float(dfr_inliers.mean())
rstd = float(dfr_inliers.std())

print("Real mean: %f\nReal std: %f" %(rmean, rstd))

# get raw data for synethetic images

dfs = pd.read_csv('synth.csv')
smean = float(dfs.mean())
sstd = float(dfs.std())

print("\nSynthetic mean: %f\nSynthetic std: %f" %(smean, sstd))

# depending on which matploblib version is running, 
# may be able to use stylesheets
# daniel really likes nicely formated plots 
# github.com/dwhit15/multiplot2d
try:
    plt.style.use("dwplot")
except:
    pass

# other style settings
colors = ("#E24A33", "#348ABD") # red, blue
data_style = dict(ls="None",marker="o",ms=5)
distribution_style = dict(lw=2)

# setup figure
plt.close("all")
plt.figure(1,dpi=100,figsize=(5,4))

# plot raw data for real and synethic images
plt.hist(np.array(dfs).flatten(), 100, normed=1, facecolor=colors[1], alpha=0.75, label="Synthetic Image Feature Data")
plt.hist(np.array(dfr).flatten(), 100, normed=1, facecolor=colors[0], alpha=0.75, label="Real Image Feature Data")

# plot modesl that fit data
x=np.linspace(0.,1.,255)
plt.plot(x,mlab.normpdf(x, smean, sstd),color=colors[1],lw=3,label="Synthetic Image Feature Model")
plt.plot(x,mlab.normpdf(x, rmean, rstd),color=colors[0],lw=3,label="Real Image Feature Model")
plt.legend()

plt.ion()
plt.grid()
plt.xlim(0,0.25)
plt.xlabel("Feature Value")
plt.ylabel("Probabilty of Feature Value")
#plt.ylim(0,0.045)
plt.show()

plt.savefig("model_distributions.png",dpi=300,transparent=False,format="png")