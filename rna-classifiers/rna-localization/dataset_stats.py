import os, sys, argparse
import pandas as pd

import util.plot as plot

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-filePath') # filepath of the raw dataset
    parser.add_argument('-column') # column to get the stats - column name or number
    parser.add_argument('-columnSeparator') # opt - text separator into a row - e.g. " "
    parser.add_argument('-label') # opt - label column - column name or number
    args = parser.parse_args()
    print("Script runs with these arguments: ", args) 

    print("Reading " +args.filePath+ " dataset...")
    dataset = pd.read_csv(args.filePath, sep='\t')

    # Get examples from column
    examples = pd.DataFrame()
    if args.column.isdigit():
        examples["data"] = dataset.iloc[:, int(args.column)]
    else:
        examples["data"] = dataset[args.column]
    
    # Label count
    if args.label:
        print("Label count:")
        if args.column.isdigit():
            print(dataset.iloc[:,int(args.label)].value_counts())
        else:
            print(dataset[args.label].value_counts())
    
    # Get string/token length
    if args.columnSeparator:
        #examples["length"] = examples.data.apply(lambda x: len(x.split(args.columnSeparator)))
        examples["length"] = examples.data.apply(lambda x: len(str(x).split(args.columnSeparator)))
    else:
        examples["length"] = examples.data.apply(lambda x: len(x))
    
    print("\nLength stats:")
    print(examples.length.describe(percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9]))

    print("\nWrite Histogram...")
    plot.histogram(data=examples.length.values, 
        bins=int(examples.length.max()/100), 
        fileName=os.path.splitext(args.filePath)[0].replace("/","_")+"_"+args.column+"_length_stats"
        )

