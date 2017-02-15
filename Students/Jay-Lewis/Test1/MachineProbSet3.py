
__author__ = 'Justin'

import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
import math

# Get Data

sample_df = pd.DataFrame.from_csv("Data1Set3_Jay-Lewis.csv")

array = sample_df.get_values()

# Decision Rule

pi0 = 0.5
pi1 = 0.5
LLR = 0.0

#Parameters H0
mu0 = [0.0, 0.0]
matrix0 = [[1.0, 0.0], [0.0, 1.0]]


#Parameters H1
mu1 = [0.0, 0.0]
value = math.sqrt(3.0)/2.0
matrix1 = [[1.0, value], [value, 1.0]]


threshold = np.log(pi0/pi1)
H = []

for observation in array:
    Ph0 = multivariate_normal.pdf(observation, mean=mu0, cov=matrix0)
    Ph1 = multivariate_normal.pdf(observation, mean=mu1, cov=matrix1)
    LLR = (np.log(Ph1) - np.log(Ph0))
    if(LLR >= threshold):
        hypothesis = 1
    else:
        hypothesis = 0

    H.append(hypothesis)

H = pd.DataFrame({'Y':H})
H.to_csv("Answer3.csv")