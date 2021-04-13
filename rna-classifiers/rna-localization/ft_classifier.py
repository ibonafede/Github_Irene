import os, sys, argparse
import numpy as np
import pandas as pd

import util.fast_text as ft
import util.plot as plot
from fastText import load_model

from sklearn.metrics import confusion_matrix, roc_curve

def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.5f}".format(1, p))
    print("R@{}\t{:.5f}".format(1, r))

def model_info(model):
    print("Dimension: ", model.get_dimension())
    print("Words: ", len(model.get_words()))
    print("Labels: ", len(model.get_labels()))

'''
# Label reference
labels_ref = {}
for i in range(0, len(labels)):
    labels_ref[i] = labels[i]
labels_ref_invert = {val:key for (key, val) in labels_ref.items()}
'''

def get_predictions(model, filePath):
    predictions = []

    # Get data
    test_set = pd.read_csv(filePath, sep='\t')
    test_set.columns = ['label', 'input']
    labels = test_set.label.unique().tolist()
    labels.sort()
    
    # Create prediction table
    for index, row in test_set.iterrows():
        label, rate = model.predict(row['input'], k=len(labels)+1)

        pred_row = {}
        pred_row['true'] = row['label']
        pred_row['predicted'] = label[0]

        for l, r in zip(label, rate):
            pred_row[l] = r
        
        predictions.append(pred_row)
    
    return pd.DataFrame(predictions), labels

def print_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Labels index: ", labels)
    print("Count\n", cm)
    print("Percentage\n",cm / cm.astype(np.float).sum(axis=0))

def plot_roc(y_true, y_preds, labels, modelPath):
    print("Labels: ", labels)
    fpr = dict()
    tpr = dict()

    for l in labels:
        y_true_temp = [1 if x==l else 0 for x in y_true]
        y_pred_temp = y_preds[l]
        fpr[l], tpr[l], _ = roc_curve(y_true_temp, y_pred_temp, pos_label=1)
        
    plot.roc(fpr, tpr, modelPath+"_roc")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode') # 'train' or 'test'
    parser.add_argument('-model') # model file path
    parser.add_argument('-filePath') # file to train or test
    args = parser.parse_args()
    print("Script runs with these arguments: ", args)

    if args.mode == 'train':
        print("\nTraining...")
        model = ft.train_supervised(args.filePath, 
            epoch=50, 
            lr=1.0,
            ws=8,
            wordNgrams=4, 
            verbose=2,
            minCount=1,
            thread=10,
            dim=5, #50
            label=u'__label__',
            pretrainedVectors="model/rnasequences2vec.vec"
            )
        
        print("\nSaving model...")
        model.save_model(args.model)
    
    else:
        print("\nLoading model...")
        model = load_model(args.model)
        model_info(model)

        print("\nTest")
        print_results(*model.test(args.filePath, k=1))
        predictions, labels = get_predictions(model, args.filePath)

        print("\nConfusion matrix")
        print_confusion_matrix(predictions.true.tolist(), predictions.predicted.tolist(), labels)

        print("\nRoc")
        plot_roc(predictions.true.tolist(), predictions[labels].to_dict('list'), labels, 
            os.path.splitext(args.model)[0].replace("/","_") + "_" + os.path.splitext(args.filePath)[0].replace("/","_"))