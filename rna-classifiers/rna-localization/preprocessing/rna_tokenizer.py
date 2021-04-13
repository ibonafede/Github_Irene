""" Create token starting RNA text sequence"""

import re

def window(sequence, window_size=5):
    """ Split sequence by sliding window
    """
    tokens = []

    for i in range(0, len(sequence)-window_size-1):
        tokens.append(sequence[i:i+window_size])

    return tokens

def different_char(sequence):
    """ Split sequence by different characters
    """
    tokens = []
    token_buffer = sequence[0]

    for c in sequence[1:]:
        if token_buffer[len(token_buffer)-1] != c:
            tokens.append(token_buffer)
            token_buffer = ""
        token_buffer += c

    return tokens

def pattern(sequence, pattern):
    """ Split sequence by different char in pattern - P.s.: len(sequence) == len(pattern)
    """
    tokens = []
    token_buffer = sequence[0]

    for i in range(1, len(pattern)):
        if pattern[i] != pattern[i-1]:
            tokens.append(token_buffer)
            token_buffer = ""
        token_buffer += sequence[i]

    return tokens

def fixed(sequence, size=5):
    """ Split sequence every 'size' char
    """
    return [sequence[i:i+size] for i in range(0, len(sequence), size)]

def delimiter(sequence, char, keep_char=True):
    if keep_char:
        return re.split("("+char+")", sequence)
    else:
        return sequence.split(char)

def merge(tokens_left, tokens_right):
    """ Merge two token sequence with the same length
    """
    tokens = []

    for l, r in zip(tokens_left, tokens_right):
        tokens.append(l)
        tokens.append(r)

    return tokens