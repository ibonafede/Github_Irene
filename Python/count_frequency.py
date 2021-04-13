#the script counts kmer frequency
#usage python count_frequency -f filename -n lenght_kmers
#output dataset_frequencies_"lenght_kmers"
#pip install scikit-bio Py3
from skbio import Sequence
import pandas as pd
import csv
import sys
import getopt
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
alphabet=Sequence(''.join(list(set(bear_to_qbear.values()))))
print(alphabet)
def func(argv):
    #print(argv)
    #print(getopt.getopt(argv,'f:n:'))
    try:
        opts, args = getopt.getopt(argv,'f:n:')
        print(opts)
    except getopt.GetoptError:
        print('Error in syntax')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-f"):
            filename = str(arg)
        elif opt in ("-n"):
            k=int(arg)
    return filename,k

def set_motif(alphabet,n):
    kmerlst=[]
    for kmer in alphabet.iter_kmers(int(n), overlap=True):
        kmerlst.append(str(kmer))
    return kmerlst

def count_kmer(string,n):
    seq=Sequence(string)
    kmerlst=[]
    freqs = seq.kmer_frequencies(int(n), relative=True, overlap=True)
    return freqs


#filename='/home/utente/irene/work/rna_localization/dataset-rnaLocalization-HS-MU.csv'
if __name__ == "__main__":
    filename,n=func(sys.argv[1:])
    data=pd.read_table(filename,header=0)

    print(data['label'].isnull().sum())
    #print(data.head())
    freq_dict_ss=data['qbearlist'].apply(count_kmer,n=n)
    freq_dict_ps=data['seqlist'].str.lower().apply(count_kmer,n=n)
    freq_df_ps=pd.DataFrame(list(freq_dict_ps)).fillna(value=0)
    freq_df_ss=pd.DataFrame(list(freq_dict_ss)).fillna(value=0)
    freq_df_ss['label']=data['label'].astype(int)
    tot=pd.concat([freq_df_ps,freq_df_ss],axis=1)
    outname='{}_{}'.format('dataset_frequencies_p&s_',str(n))
    tot.round(decimals=3).to_csv(outname,quoting=csv.QUOTE_NONE,header=True,index=False,sep='\t')
