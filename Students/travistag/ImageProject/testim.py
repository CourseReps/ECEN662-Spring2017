from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def normalizeHisto(h):
	tot = sum(h)
	ret = []
	for v in h:
		ret.append(v/tot)
	return ret


rim = Image.open('TrainingSetScenes/1.jpg')
sim = Image.open('TrainingSetSynthetic/image1.png')

rhist = normalizeHisto(rim.convert(mode= 'L').histogram())
shist = normalizeHisto(sim.convert(mode= 'L').histogram())
plt.figure(1)
plt.plot(np.arange(len(rhist)), rhist)

plt.figure(2)
plt.plot(np.arange(len(shist)), shist)
plt.show()

# realvals = []
# synthvals = []

# for filename in os.listdir('./TrainingSetScenes/'):
# 	cimage = Image.open('./TrainingSetScenes/'+str(filename))
# 	cimage = cimage.convert(mode = 'L')
# 	chist = normalizeHisto(list(cimage.histogram()))
# 	realvals.append(max(chist))
# 	print('R')

# for filename in os.listdir('./TrainingSetSynthetic/'):
# 	cimage = Image.open('./TrainingSetSynthetic/'+str(filename))
# 	cimage = cimage.convert(mode = 'L')
# 	chist = normalizeHisto(list(cimage.histogram()))
# 	synthvals.append(max(chist))
# 	print('S')

# df = pd.DataFrame(realvals)
# dfs = pd.DataFrame(synthvals)

# df.to_csv('real.csv', index = False)
# dfs.to_csv('synth.csv', index = False)
