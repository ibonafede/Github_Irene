import csv
import pandas as pd

qbear_map = {
  'Z': ['a', 'b', 'c', 'd', 'e'],
  'A': ['f', 'g', 'h', 'i'],
  'Q': ['='],
  'X': ['j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r'],
  'S': ['s', 't', 'u', 'v', 'w', 'x', 'y', 'z'],
  'W': ['^'],
  'C': ['!', '\"', '#','$','%','2','3','4','5','6'],
  'D': ['&', '\'', '(', ')', '7', '8', '9', '0'],
  'E': ['+', '>'],
  'B': ['[', ']'],
  'G': ['{', '}'],
  'T': [':'],
  'V': ['A', 'B', 'C', 'D', 'E'],
  'F': ['F', 'G', 'H', 'I'],
  'R': ['J'],
  'N': ['K', 'L', 'M', 'N', 'Y', 'Z', '~', '?'],
  'H': ['O', 'P', 'Q', 'R', 'S', '_', '/', '\\'],
  'Y': ['T', 'U', 'W', 'Y', 'Z', '@']
}

bear_to_qbear = { '9': 'D', 'S': 'H', '0': 'D', 's': 'S', '7': 'D', '&': 'D', 'D': 'V', 
  '\\': 'H', 'F': 'F', 'P': 'H', 'R': 'H', 'd': 'Z', 'U': 'Y', 'i': 'A', 't': 'S', 'x': 'S', 
  'y': 'S', ':': 'T', 'o': 'X', '%': 'C', 'Y': 'N', 'A': 'V', "'": 'D', 'q': 'X', 'I': 'F', 
  'r': 'X', 'B': 'V', 'l': 'X', 'k': 'X', 'h': 'A', 'v': 'S', 'N': 'N', 'T': 'Y', '4': 'C', 
  '5': 'C', 'O': 'H', '}': 'G', '~': 'N', '_': 'H', ')': 'D', 'J': 'R', '>': 'E', 'w': 'S', 
  'p': 'X', 'C': 'V', 'c': 'Z', '^': 'W', 'g': 'A', 'a': 'Z', 'b': 'Z', 'm': 'X', 'M': 'N', 
  '?': 'N', '(': 'D', 'e': 'Z', 'K': 'N', 'W': 'Y', 'Z': 'N', '[': 'B', '+': 'E', '@': 'Y', 
  '!': 'C', '{': 'G', '"': 'C', '3': 'C', 'L': 'N', 'z': 'S', '/': 'H', 'G': 'F', 'Q': 'H', 
  ']': 'B', '2': 'C', 'n': 'X', '#': 'C', 'H': 'F', '$': 'C', 'u': 'S', 'E': 'V', 'f': 'A', 
  '6': 'C', '8': 'D', 'j': 'X', '=': 'Q'
  }

def to_qbear(string_bear_format):
  string_qbear_format = ""

  for c in string_bear_format:
    if c in bear_to_qbear:
      string_qbear_format += bear_to_qbear[c]
    else:
      string_qbear_format += c
      print("Warning: char '" + c + "' does not converted")

  return string_qbear_format