import pandas as pd
import scipy.stats
import sys
import os
from PIL import Image


def normalizeHisto(h):
	tot = sum(h)
	ret = []
	for v in h:
		ret.append(v/tot)
	return ret

def classifyImage(m0, s0, m1, s1, img):
	img = img.convert(mode = 'L')
	val = max(normalizeHisto(list(img.histogram())))
	l0 = scipy.stats.norm.pdf(val, m0, s0)
	l1 = scipy.stats.norm.pdf(val, m1, s1)
	if(l0>l1):
		return 0
	else:
		return 1


dfr = pd.read_csv('real.csv')
dfs = pd.read_csv('synth.csv')
rmean = float(dfr.mean())
smean = float(dfs.mean())
rstd = float(dfr.std())
sstd = float(dfs.std())


if(len(sys.argv)<2):
	print('Need a directory to predict')
	exit()
direct = sys.argv[1]
predictions = []


for filename in os.listdir(direct):
	im = Image.open(str(direct)+str(filename))
	predictions.append(classifyImage(rmean, smean, rstd, sstd, im))

df = pd.DataFrame(predictions)
df.to_csv('predictions.csv', index = False)