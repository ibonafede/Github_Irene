import os, sys, argparse, csv
import numpy as np
import pandas as pd

import preprocessing.rna_tokenizer as tokenizer
import preprocessing.rna_vectorizer as vectorizer
import util.split as util_split

def get_dataset_raw(filePath):
    """ Load dataset """
    dataset = pd.read_csv(filePath, sep='\t')

    # If dataset is known, filter it
    fileName = os.path.splitext(os.path.basename(filePath))[0]
    if fileName == 'lnc_rna_localization_2602':
        list_length = dataset.seqlist.apply(lambda x: len(x)) # Get lenght of primary sequence
        dataset = dataset[list_length<=10000] # Delete all long sequences
        dataset = dataset[dataset.label != 2] # Delete label=2
        dataset = dataset[dataset.species != 'cell'] # Delete species='cell'
        dataset = dataset.reset_index(drop=True)
        dataset.label = dataset.label.astype(int)
    elif fileName == 'nc_rna_2602':
        pass # No op
    
    return dataset

def adjust_length_sequences(dataset, column_left, column_right):
    dataset_new = dataset[dataset[column_left].str.len() == dataset[column_right].str.len()]
    if dataset.shape[0] != dataset_new.shape[0]:
        dataset = dataset.reset_index(drop=True)
        print("WARNING: dataset was changed because contains rows with different len sequences")
        return dataset_new
    else:
        return dataset

def preprocessing(dataset_raw, name, process_input):
    """ Dataset preprocessing """

    dataset = None

    if name == 'token_window':
        # Input: [column, number]
        column = process_input[0]
        number = int(process_input[1])
        dataset = dataset_raw[column].apply(lambda x: ' '.join(tokenizer.window(x, number)))
    elif name == 'token_different_char':
        # Input: [column]
        column = process_input[0]
        dataset = dataset_raw[column].apply(lambda x: ' '.join((tokenizer.different_char(x))))
    elif name == 'token_pattern':
        # Input: [column, columnPattern]
        column = process_input[0]
        columnPattern = process_input[1]
        dataset_raw = adjust_length_sequences(dataset_raw, column, columnPattern)
        dataset = dataset_raw.apply(lambda x: ' '.join(tokenizer.pattern(x[column], x[columnPattern])), axis=1)
    elif name == 'token_fixed':
        # Input: [column, number]
        column = process_input[0]
        number = int(process_input[1])
        dataset = dataset_raw[column].apply(lambda x: ' '.join(tokenizer.fixed(x, number)))
    elif name == 'token_delimiter':
        # Input: [column, character, keep]
        column = process_input[0]
        character = process_input[1]
        keep = bool(process_input[2])
        dataset = dataset_raw[column].apply(lambda x: ' '.join(tokenizer.delimiter(x, character, keep)))
    elif name == 'token_window_mixed':
        # Input: [column1, column2, number]
        column1 = process_input[0]
        column2 = process_input[1]
        number = int(process_input[2])
        dataset_raw = adjust_length_sequences(dataset_raw, column1, column2)
        dataset = dataset_raw.apply(lambda x: ' '.join(tokenizer.merge(tokenizer.window(x[column1], number), tokenizer.window(x[column2], number))), axis=1)
    elif name == 'vec_one_hot':
        # Input: [column]
        column, tokens = process_input[0], []
        dataset_raw[column].apply(lambda x: tokens.extend(list(filter(None, x.split(" ")))))
        _, char_to_int, int_to_char = vectorizer.get_alphabet(tokens)
        print("Alphabet: ", char_to_int, int_to_char)
        dataset = dataset_raw[column].apply(lambda x: ' '.join(vectorizer.one_hot_label(x.split(" "), char_to_int)))
    
    return pd.DataFrame.from_dict({'example': dataset.tolist()})

def insert_first_column(dataset_raw, dataset, first_column, first_column_prefix):
    """ Insert label """
    dataset.insert(0, "label", dataset_raw[first_column])

    # Add prefix
    if first_column_prefix:
        dataset.label = dataset.label.apply(lambda x: first_column_prefix+str(x))
    
    return dataset

def insert_auxiliaries_columns(dataset_raw, dataset, columns):
    dataset_raw_filtered = dataset_raw.ix[dataset.index.tolist()]
    for column in columns:
        dataset = pd.concat([dataset, dataset_raw_filtered[column]], axis=1)
    return dataset

def delete_auxiliaries_columns(dataset, columns):
    for column in columns:
        del dataset[column]
    return dataset

def split(dataset, stratify, test_size=0.20, random_state=42):

    # Split - training_set_full, test_set
    print("\nCreate training_set_full & test_set")
    training_set_full, test_set = util_split.base_train_test(dataset=dataset, stratify=stratify, test_size=test_size, random_state=random_state)

    # Split - training_set, valid_set
    print("\nCreate training_set & valid_set")
    training_set, valid_set = util_split.base_train_test(dataset=training_set_full, stratify=stratify, test_size=test_size, random_state=random_state)

    return training_set, valid_set, test_set

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-filePrefix') # name
    parser.add_argument('-dataset') # filepath of the raw dataset with header
    parser.add_argument('-preprocName') # function
    parser.add_argument('-preprocInput') # one or more columns of the dataset - e.g. col1,col2,num or col2
    parser.add_argument('-firstColumn') # opt - e.g. label or id
    parser.add_argument('-firstColumnPrefix') # opt - add prefix
    parser.add_argument('-split') # opt - split balanced by one or more columns of the dataset
    parser.add_argument('-splitRatio', type=float, default=0.2) # opt - default = 0.2, range = [0, 1]
    parser.add_argument('-keepHeader', type=int, default=1) # opt - keep header into new dataset
    args = parser.parse_args()
    print("Script runs with these arguments: ", args)

    print("Get raw dataset...")
    dataset_raw = get_dataset_raw(args.dataset)

    print("Preprocessing...")
    dataset = preprocessing(dataset_raw, args.preprocName, args.preprocInput.split(","))

    if args.firstColumn:
        print("Insert first column...")
        dataset = insert_first_column(dataset_raw, dataset, args.firstColumn, args.firstColumnPrefix)
    
    if args.split:
        print("Split...")
        auxiliaries_columns = [x for x in args.split.split(",") if x != args.firstColumn]
        # Insert columns for balance (if exist)
        dataset = insert_auxiliaries_columns(dataset_raw, dataset, auxiliaries_columns)
        # Split
        training_set, valid_set, test_set = split(dataset=dataset, stratify=args.split.split(","), test_size=args.splitRatio)
        # Delete columns for balance (if exist)
        dataset = delete_auxiliaries_columns(dataset, auxiliaries_columns)
        training_set = delete_auxiliaries_columns(training_set, auxiliaries_columns)
        valid_set = delete_auxiliaries_columns(valid_set, auxiliaries_columns)
        test_set = delete_auxiliaries_columns(test_set, auxiliaries_columns)

        print("Save splitted sets...")
        training_set.to_csv(args.filePrefix+'training_set.tsv', sep='\t', index=None, encoding='utf-8', header=None, quoting=csv.QUOTE_NONE)
        valid_set.to_csv(args.filePrefix+'valid_set.tsv', sep='\t', index=None, encoding='utf-8', header=None, quoting=csv.QUOTE_NONE)
        test_set.to_csv(args.filePrefix+'test_set.tsv', sep='\t', index=None, encoding='utf-8', header=None, quoting=csv.QUOTE_NONE)
        
    print("Save dataset...")
    if args.keepHeader:
        dataset.to_csv(args.filePrefix+'data_set.tsv', sep='\t', index=None, encoding='utf-8', quoting=csv.QUOTE_NONE)
    else:
        dataset.to_csv(args.filePrefix+'data_set.tsv', sep='\t', index=None, encoding='utf-8', header=None, quoting=csv.QUOTE_NONE)

    


    