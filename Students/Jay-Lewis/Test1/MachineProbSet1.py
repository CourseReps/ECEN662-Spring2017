__author__ = 'Justin'

import pandas as pd

# Get Data

sample_df = pd.DataFrame.from_csv("Data1Set1_Jay-Lewis.csv")

# Decision Rule

Y_hat = (sample_df > 0.25)
Y_hat['Y'] = Y_hat['Y'].map({False: 0, True: 1})

Y_hat.to_csv("Answer1.csv")