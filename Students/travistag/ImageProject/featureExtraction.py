from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def normalizeHisto(h):
	"""
    Normalize a 1-channel image histogram. The histogram should be passed
    as a list. 
    """
	tot = sum(h)
	ret = []
	for v in h:
		ret.append(v/tot)
	return ret




realvals = []
synthvals = []

#Collect features for all files in the scene training directory, and add to a list
for filename in os.listdir('./TrainingSetScenes/'):

	#Open image and convert to greyscale
	cimage = Image.open('./TrainingSetScenes/'+str(filename))
	cimage = cimage.convert(mode = 'L')

	#Obtain and normalize histogram of intensity values
	chist = normalizeHisto(list(cimage.histogram()))

	#Append maximum value from this normalized histogram to the list of features
	realvals.append(max(chist))

#Collect features for all files in the synthetic training directory, and add to a list
for filename in os.listdir('./TrainingSetSynthetic/'):

	#Open image and convert to greyscale
	cimage = Image.open('./TrainingSetSynthetic/'+str(filename))
	cimage = cimage.convert(mode = 'L')

	#Obtain and normalize histogram of intensity values
	chist = normalizeHisto(list(cimage.histogram()))

	#Append maximum value from this normalized histogram to the list of features
	synthvals.append(max(chist))

#Create dataframes for simple csv output
df = pd.DataFrame(realvals)
dfs = pd.DataFrame(synthvals)

#Output values to csv, which can be opened into dataframes to obtain mean, std deviation, etc.
df.to_csv('real.csv', index = False)
dfs.to_csv('synth.csv', index = False)
