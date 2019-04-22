import pandas as pd


#treating missing Data
#drop_na,exclude labels from a data set which refer to missing data
df=df.dropna(subset=col)
#delete all rows
df.dropna(axis=0)
#delete all columns
df.dropna(axis=0)
#fillna
df.fillna(0)
#fill a DataFrame with the mean of that column
df.fillna(df.mean())



