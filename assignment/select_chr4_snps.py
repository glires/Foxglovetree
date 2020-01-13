#!/usr/bin/env python

# select_chr4_snps.py
# coded by Kohji

import sys

omni = {}	# RefSNP and chr4_position (GRCh37)
with open('GPL16104_chr4.tsv') as gpl16104:
  for line in gpl16104:
    fields = line[:-1].split("\t")
    omni[fields[0]] = fields[2]

#### format of GPL16104_chr4.tsv
## rs4974674	4	2311361
## rs4479771	4 	2316874
## rs654207	4	2324865
## rs2097298	4	2329295

for line in sys.stdin:
  fields = line.split("\t")
  if len(fields) > 8:	# to exclude header lines
    if fields[2] in omni:	# if in HumanOmni2.5-8 chr4
      if omni[fields[2]] == fields[1]:	# chromosomal position
        if len(fields[3]) == 1:	# single nucleotide
          if len(fields[4]) == 1:	# single nucleotide
            if fields[5] == '100':	# QUAL
              if fields[6] == 'PASS':	# 38738 entries
                if int(fields[2][2:]) % 9 == 0:	# to reduce data
                  print(line, end = '')

#### format of genotypes.vcf
## 4	134851  rs3829	T	C	100	PASS	AC=743;
## 4	134861	rs550591365	T	C	100	PASS	AC=1;
## 4	134978	rs568721674	T	G	100	PASS	AC=1;
