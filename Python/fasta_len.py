#usage: python FastaFile min_length max_length
#the script parse a fasta file by sequences length
import sys
from Bio import SeqIO


def parse_fasta_by_len(FastaFile,min_length,max_length):
	output_file=open('{}_{}_{}'.format(FastaFile,str(min_length),str(max_length)),'w')
	output_file_exception=open('{}_{}{}'.format(FastaFile.strip('.fa'),'exception','.fa'),'w')
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

if __name__='__main__':
	FastaFile = sys.argv[1]
	#max_length_Vienna=91670
	min_length=sys.argv[2]
	#200 is min seq len for lncRNAs
	max_length=sys.argv[3]
	parse_fasta_by_len(FastaFile,min_length,max_length)
