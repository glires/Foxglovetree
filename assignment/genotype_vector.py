#!/usr/bin/env python

# genotype_vector.py

import sys

idividuals = {}	# e.g., key: NA19058; value: JPT
with open('sample_info.tsv') as info:
  for line in info:
    fields = line[:-1].split("\t")
    idividuals[fields[0]] = fields[1]

column_number = {}	# e.g., key: NA21144; value 2512
with open('individuals.tsv') as inds:
  fields = (inds.readline())[:-1].split("\t")
  for i in range(len(fields)):
    column_number[fields[i]] = i

with open('genotypes.vcf') as gt:

  print('ID', end = '')	# print the header line
  for line in gt:	# print the header line
    fields = line.split("\t")
    print("\t" + fields[2], end = '')
  print()

  for id in idividuals.keys():
    if id in column_number:
      cn = column_number[id]
    else:	# no genotype data
      sys.stderr.write('KeyError: ' + id + "\n")
      continue
    print(id + '_' + idividuals[id], end = '')

    gt.seek(0)
    for line in gt:
      fields = line[:-1].split("\t")
      if (fields[cn]) == '0|0':
        print("\t0.0", end = '')
      elif (fields[cn]) == '1|1':
        print("\t1.0", end = '')
      else:	# hetero and others
        print("\t0.5", end = '')
    print()
