#the script extract RNA sequences from Homo_sapiens.GRCh38.ncrna_new_header.fa
#fold RNA by RNAfold package
#convert dot and bracket into BEAR

#filename is a file with the RNAs ID (gene name or ENSEMBL id gene as well transcript)
#eg python -f prova -s HS


#import modules
import sys,os
from Bio import SeqIO
import pandas as pd
import getopt
import collections
import re
import csv


def func(argv):
    print(argv)
    print(getopt.getopt(argv,'f:s:'))
    try:
        opts, args = getopt.getopt(argv,'f:s:w:')
        print(opts)
    except getopt.GetoptError:
        print('Error in syntax')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-f"):
            filename = str(arg)
        elif opt in ("-s"):
            species = str(arg)

        #elif opt in ("-w"):
            #outname = str(arg)
    return filename,species

def parse_fasta_by_len(FastaFile,min_length,max_length,outname):
    #retrieve sequences from 200 to 91669
    #max_length_Vienna=91670
	min_length=200
	#200 is min seq len for lncRNAs
	max_length=91669
	output_file=open(outname,'w')
	output_file_exception=open('{}_{}{}'.format(FastaFile.strip('.fasta'),'exception','.fa'),'w')
	fasta_sequences = SeqIO.parse(open(FastaFile),'fasta')
	for fasta in fasta_sequences:
		rec, seq = fasta.id, str(fasta.seq)
		#print(seq)
		if (len(seq)<int(min_length) or len(seq)>int(max_length)):
			output_file_exception.write('>'+rec+'\n'+seq+'\n')
			print(rec,len(seq))
		if (len(seq)>int(min_length) and len(seq)<int(max_length)):
			print(len(seq))
			output_file.write('>'+rec+'\n'+seq+'\n')

def fold(outname):
    os.system('cat '+outname+ ' | RNAfold --noPS > outVienna.fa')

def runBear():
    #delete energy
	cmd0="cat outVienna.fa | awk '{print $1}' > tmp0.fa"
	print(cmd0)
	os.system(cmd0)
	infile=open(filename,'r')
	start=1
	end=len(infile.readlines())
	print(end)
	for i in range(start,end,3):
		print(i,i+2)
        #write one sequence file
		cmd=("sed -n "+str(i)+","+str(i+2)+"p "+filename+" > output/tmp/tmp.fa")
        os.system(cmd)
        #ruun Bear
		cmd3="java -jar BearEncoder.new.jar output/tmp/tmp.fa"+" output/tmp/Out_Bear"+str(i)+'.fb'
		print(cmd3)
		os.system(cmd3)
        #append new sequence to outBear.fb
        cmd_append="cat output/tmp/Out_Bear"+str(i)+".fb >> outBear.fb"
        os.system(cmd_append)
        #clean folder
        cmd_remove=os.system("rm output/tmp/Out_Bear"+str(i)+".fb")
        os.system(cmd_remove)
        print(cmd_append,'\n',cmd_remove)

def validate(seq):
    lenseqlist=[]
    lenbearlist=[]
    lendotlist=[]
    alphabets = {'dna': re.compile('^[acgtn]*$', re.I),'rna': re.compile('^[acgun]*$', re.I), 'protein': re.compile('^[acdefghiklmnpqrstvwy]*$', re.I),'dotbracket':re.compile("^['(','.']")}
    if alphabets['rna'].search(seq.strip('\n')) is not None:
        seqlist.append(seq.strip('\n'))
        lenseqlist.append(len(seq))
    if alphabets['dotbracket'].search(seq.strip('\n')) is not None:
        dotlist.append(seq.strip('\n'))
        lendotlist.append(len(seq))
    if alphabets['rna'].search(seq.strip('\n')) is None:
        if alphabets['dotbracket'].search(seq.strip('\n')) is None:
            bearlist.append(seq.strip('\n'))
            lenbearlist.append(len(seq))
    return seqlist,dotlist,bearlist

def get_label(data):
    #data=pd.read_table(f,sep='\t',header=0)
    l=data['ID'].str.split('_'+species+'_',expand=True)[1].str[0]
    data['label']=l
    #outname='_'.join(['dataset',species])
    data['species']=species
    #data[data['label']!=2]
    #data[['label','species','seqlist','dotlist','bearlist','ID']].to_csv(outname,header=True,index=None,sep='\t',quoting=csv.QUOTE_NONE)
    return data,species,filename

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

  return string_qbear_format


def remove_diffLenSeqs(filename):
  data=read_table(filename,'\t',header=0)
  data.head()




if __name__='__main__':
    filename,species=func(sys.argv[1:]
    fileSeq='{}_{}{}'.format('retrieveSeq',species,'.sh')
    os.system('bash  '+fileSeq+' '+filename)
    #output is out.fasta
    FastaFile = 'out.fasta'
    outname='{}_{}_{}'.format(FastaFile.strip('.fasta'),str(min_length),str(max_length))
	parse_fasta_by_len(FastaFile,outname)
    fold(outname)
    #output outVienna.fa
    #convert to Bear format
    runBear()
    #output outBear.fb
    #fromfb2dataset
    seqlist=[]
    dotlist=[]
    bearlist=[]
    ID=[]
    ID_rnacentral=[]
    #from Fasta file 2 dataframe
    for line in open(filename,'r'):
        if not line.startswith('>'):
            seqlist,dotlist,bearlist=validate(line.strip('\n'))
        elif line.startswith('>'):
            ID.append(line.strip('\n').strip('>'))
    dataframe=pd.DataFrame({'ID':ID[0:len(seqlist)],'seqlist':seqlist[0:len(seqlist)],'dotlist':dotlist[0:len(seqlist)],'bearlist':bearlist[0:len(seqlist)]})
    dataframe=dataframe.reset_index(drop=True)
    dataframe=dataframe.drop_duplicates()
    #dataframe.to_csv(outname,sep='\t',index=None,header=True,quoting=csv.QUOTE_NONE)
    data,species,filename=get_label(dataframe)
    data['qbearlist'] = data.bearlist.apply(lambda x: to_qbear(x))
    outname='_'.join(['dataset',species])
    data=data.dropna(subset=['label'])
    data[data.seqlist.str.len()!=data.bearlist.str.len()].to_csv('seq.diff.len',header=True,index=None,sep='\t',quoting=csv.QUOTE_NONE)
    data[data.seqlist.str.len()==data.bearlist.str.len()].to_csv('seq.equal.len',header=True,index=None,sep='\t',quoting=csv.QUOTE_NONE)
    print('species\n',species,'\n',data['label'].value_counts())
    data[['ID','label','species','seqlist','dotlist','bearlist','qbearlist']].to_csv(outname,header=True,index=None,sep='\t',quoting=csv.QUOTE_NONE)


