import pandas as pd
import getopt
import matplotlib.pyplot as pyplot
def func(argv):
    #get argv as input
    #print(getopt.getopt(argv,'f:s:w:'))
    try:
        opts, args = getopt.getopt(argv,'f:s:w:')
        print(opts)
    except getopt.GetoptError:
        print('Error in syntax')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-f"):
            filename = str(arg)
    return filename

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
#linear interpolation at missing datapoints

#interpolate functions

#If you are dealing with a time series that is growing at an increasing rate, method='quadratic' may be appropriate.
#If you have values approximating a cumulative distribution function, then method='pchip' should work well.
#To fill missing values with goal of smooth plotting, use method='akima'.
#spline : df.interpolate(method='spline', order=2)

def interpolation(df):
    methods = ['linear', 'quadratic', 'cubic']
    df = pd.DataFrame({m: ser.interpolate(method=m) for m in methods})
    #show plot
    df.plot()
