#!/bin/bash

# Supervised - seqlist (pattern bear), label and species balanced
#python3 dataset_create.py -filePrefix data/supervised/seqlist_ -dataset data/raw/lnc_rna_localization_2602.tsv -preprocName token_pattern -preprocInput seqlist,bearlist -firstColumn label -firstColumnPrefix __label__ -split label,species -splitRatio 0.2

# Doc2Vec Unsupervised - seqlist (pattern bear)
#python3 dataset_create.py -filePrefix data/unsupervised/seqlist_ -dataset data/raw/nc_rna_2602.tsv -preprocName token_pattern -preprocInput seqlist,bearlist -firstColumn id -firstColumnPrefix __label__ -keepHeader 0

# Supervised - seqlist, label and species balanced - Vectorized
python3 dataset_create.py -filePrefix data/supervised/seqlist_vec_ -dataset data/raw/lnc_rna_localization_2602.tsv -preprocName token_fixed -preprocInput seqlist,1 -firstColumn label -firstColumnPrefix __label__
python3 dataset_create.py -filePrefix data/supervised/seqlist_vec_ -dataset data/supervised/seqlist_vec_data_set.tsv -preprocName vec_one_hot -preprocInput example -firstColumn label