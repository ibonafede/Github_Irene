import os
import sys
import pandas as pd
import pickle
from Bio import SeqIO

#input_file='/home/bonafede/lavoro/genome/hg19/Homo_sapiens.GRCh37.75.ncrna_m.fa'
#input_file='/home/bonafede/lavoro/CRG_VivoVitro/dati/slide_sequences_20_06_2017.U.slideseq.fa'
def main(input_file):
	fasta_sequences = SeqIO.parse(open(input_file),'fasta')
	g=[]
	t=[]
	t_type=[]
	t_biotype=[]
	g_type=[]
	chrm=[]
	start=[]
	s=[]
	n=[]
	gene=[]
	for fasta in fasta_sequences:
		name, sequence = fasta.id, fasta.seq.tostring()
		#name2=name.split()[-1]
		n.append(name)
		s.append(sequence)
		#gene.append(name2)
	df=pd.DataFrame({'name':n,'sequences':s})
	print(df.head())
	return df

if __name__ == "__main__":
    main(input_file)
