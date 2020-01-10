#!/usr/bin/env python

# grep_one_chr.py - grep RefSNP data on one specified human chromosome
# coded by Kohji

import sys

if len(sys.argv) < 3:
  sys.stderr.write('Usage: grep_one_chr.py 18 GPL16104-1240.txt')
  sys.stderr.write('    # if chr18' + "\n")
  sys.exit(82)

with open(sys.argv[2]) as data_snp:
  for line in data_snp:
    fields = line[:-1].split("\t")
    if len(fields) >= 4:
      if fields[1] == sys.argv[1]:
        if fields[0] == fields[3]:
          if fields[0][0:2] == 'rs':
            print(fields[0], fields[1], fields[2], sep = "\t")
