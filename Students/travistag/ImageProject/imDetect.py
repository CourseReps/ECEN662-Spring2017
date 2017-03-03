import pandas as pd
import scipy.stats
import sys
import os
from PIL import Image


def normalizeHisto(hist_list):
    """
    Normalize a 1-channel image histogram. The histogram should be passed
    as a list. 
    """
    total = sum(hist_list)
    return [v*1./total for v in hist_list]

def classifyImage(m0, s0, m1, s1, img):
    """
    Clasify an image.
    """
    # convert image to monochrome
    img = img.convert(mode = 'L')
    # calculate a luminance histogram 
    # normalize it so that images of different sizes can be compared
    # then get the normalized occurence of the most prevalent intensity
    val = max(normalizeHisto(list(img.histogram())))
    # calculate the likelihood of this intensity given hypothesis 0
    l0 = scipy.stats.norm.pdf(val, m0, s0)
    # calculate the likelihood of this intensity given hypothesis 1
    l1 = scipy.stats.norm.pdf(val, m1, s1)
    # assuming equal priors, make a decision
    if(l0>l1):
        return 0
    else:
        return 1

# approximate Guassian distribution for real images
rmean = 0.014482
rstd = 0.005275

# approximate Guassian distribution for synethic images
smean = 0.153275
sstd = 0.030564

# loop through all images in a directory and classify them
if(len(sys.argv)<2):
	print('Need a directory to predict')
	exit()
direct = sys.argv[1]
predictions = []
for filename in os.listdir(direct):
	im = Image.open(str(direct)+str(filename))
	predictions.append(classifyImage(rmean, rstd, smean, sstd, im))

# save predictions to file
df = pd.DataFrame(predictions)
df.to_csv('predictions.csv', index = False)

print(len(predictions))
print(sum(predictions))