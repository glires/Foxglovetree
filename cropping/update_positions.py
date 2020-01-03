#!/usr/bin/env python

# update_positions.py
# coded by Kohji

import sys

if len(sys.argv) < 3:
  sys.stderr.write("Usage: update_positions.py old.tsv add.tsv\n")
  sys.exit(82)

data = {}

with open(sys.argv[2]) as new:
  for line in new:
    fields = line[:-1].split("\t")
    data[fields[0]] = line

with open(sys.argv[1]) as old:
  for line in old:
    fields = line[:-1].split("\t")
    if fields[0] in data:
      print(data[fields[0]], end = '')
    else:
      print(line, end = '')
