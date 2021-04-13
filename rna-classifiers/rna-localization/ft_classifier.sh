#!/bin/bash

# Training
python3 ft_classifier.py -mode train -model model/rna_localization.bin -filePath data/supervised/seqlist_training_set.tsv

# Validation
#python3 ft_classifier.py -mode test -model model/rna_localization.bin -filePath data/supervised/seqlist_valid_set.tsv

# Test
#python3 ft_classifier.py -mode test -model model/rna_localization.bin -filePath data/supervised/seqlist_test_set.tsv