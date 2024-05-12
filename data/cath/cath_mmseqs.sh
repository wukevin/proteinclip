#!/bin/bash
# Mean to be executed in the same CATH folder as the input fasta file for CATH s20
mkdir mmseqs_working_dir && cd mmseqs_working_dir

mmseqs createdb ../cath-dataset-nonredundant-S20-v4_3_0.fa queryDB
mmseqs search queryDB queryDB resultDB tmp -e 10000 -s 7.5 --max-seqs 300
mmseqs convertalis queryDB queryDB resultDB resultDB.m8
cd ../
mv mmseqs_working_dir/resultDB.m8 mmseqs_output.m8 
rm -r mmseqs_working_dir
