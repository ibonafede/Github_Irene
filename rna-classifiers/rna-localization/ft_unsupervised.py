import os
import sys
import pandas as pd
import util.fast_text as ft

from fastText import train_unsupervised

if __name__ == "__main__":
    dataset = "data/unsupervised/seqlist_data_set.tsv"
    model_name = "model/rnasequences2vec.bin"

    print("Train...", dataset)
    model = train_unsupervised(input=dataset,
        model='cbow',
        lr=0.01,
        dim=200,
        wordNgrams=4,
        minCount=1,
        epoch=10
    )

    print("Save...")
    model.save_model(model_name)

    print("Create .vec file...")
    ft.bin_to_vec(model_name)

