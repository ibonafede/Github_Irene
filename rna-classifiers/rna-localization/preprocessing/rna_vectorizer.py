""" Create numpy vector starting from RNA text sequence"""

def get_alphabet(tokens):

    alphabet = list(set(tokens))
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    
    return alphabet, char_to_int, int_to_char

def one_hot_label(tokens_sequence, alphabet):

    token_one_hot = []
    for token in tokens_sequence:
        token_one_hot.append(str(alphabet[token]))

    return token_one_hot
    