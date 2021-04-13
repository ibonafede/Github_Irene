#!/bin/bash

# Supervised raw
python3 dataset_stats.py -filePath data/raw/lnc_rna_localization_2602.tsv -column seqlist -label label

# Supervised seqlist_data_set preprocessed
python3 dataset_stats.py -filePath data/supervised/seqlist_data_set.tsv -column 1 -label 0 -columnSeparator " "

# Unsupervised raw
python3 dataset_stats.py -filePath data/raw/nc_rna_2602.tsv -column seqlist

# Unsupervised preprocessed
python3 dataset_stats.py -filePath data/unsupervised/seqlist_data_set.tsv -column 1 -columnSeparator " "