#!/bin/bash

$MY_ROOT_APPLICATION/Starspace/starspace train -trainFile data/unsupervised/seqlist_data_set.tsv -model model/ncrna2vec -fileFormat labelDoc -epoch 20 -dim 50 -ngrams 4 -ws 8