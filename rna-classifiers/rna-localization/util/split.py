""" Utilities """

from sklearn.model_selection import train_test_split

def base_train_test(dataset, stratify, test_size=0.20, random_state=42):
    """
    dataset: all dataset (pandas format)
    stratify: columns to balanced (string format)
    """
    training_set, test_set = train_test_split(dataset,
        test_size=test_size, random_state=random_state, shuffle=True, stratify=dataset[stratify])

    for col in stratify:
        print("Training set - "+col+".value_counts")
        print(training_set[col].value_counts())
        print("Test set - "+col+".value_counts")
        print(test_set[col].value_counts())

    return training_set, test_set