__author__ = 'Justin'

import pandas as pd
import numpy as np
from scipy.stats import chi2,ncx2,beta


# Get Data

sample_df = pd.DataFrame.from_csv("Data1Set1_Jay-Lewis.csv")

array = sample_df.get_values()

# Decision Rule

pi0 = 0.75
pi1 = 0.25
LLR = 0.0
a = 2
b = 4
hypothesis = 0
threshold = pi0/pi1
H = []

for observation in array:
    Ph0 = beta.pdf(observation, a, b)
    Ph1 = beta.pdf(observation, b, a)
    LR = Ph1/Ph0
    if(LR >= threshold):
        hypothesis = 1
    else:
        hypothesis = 0

    H.append(hypothesis)

H = pd.DataFrame({'Z':H})
gitID = "Jay-Lewis"
H.to_csv("Test1Answer1_"+gitID+".csv")