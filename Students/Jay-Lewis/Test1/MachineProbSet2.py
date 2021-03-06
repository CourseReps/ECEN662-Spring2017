
__author__ = 'Justin'

import pandas as pd
import numpy as np
from scipy.stats import chi2,ncx2

# Get Data

sample_df = pd.DataFrame.from_csv("Data1Set2_Jay-Lewis.csv")

array = sample_df.get_values()

# Decision Rule

pi0 = 0.5
pi1 = 0.5
LLR = 0.0
k0= 4
k1 = 4
lambda1 = 2
hypothesis = 0
threshold = np.log(pi0/pi1)
H = []

for observation in array:
    Ph0 = chi2.pdf(x = observation,df = k0)
    Ph1 = ncx2.pdf(x = observation, df = k1, nc = lambda1)
    LLR = (np.log(Ph1) - np.log(Ph0))
    if(LLR >= threshold):
        hypothesis = 1
    else:
        hypothesis = 0

    H.append(hypothesis)

H = pd.DataFrame({'Z':H})
gitID = "Jay-Lewis"
H.to_csv("Test1Answer2_"+gitID+".csv")