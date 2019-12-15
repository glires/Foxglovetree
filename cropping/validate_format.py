#!/usr/bin/env python3

# validate_format.py - validate the tab-delimited text file
# coded   by Kohji on 2019-06-28
# updated by Kohji on 2019-12-15

import re
import sys
import codecs	# in order to detect carriage return (CR)

number_of_fields = 6

# check the number of command arguments
if len(sys.argv) < 2:
  sys.stderr.write("Error 12: An input text file is requied as the first command argument.\n")
  sys.exit(12)

# check byte order mark (BOM)
tsv = open(sys.argv[1], encoding = 'utf-8')
line = tsv.readline()
if line[0] == '\ufeff':
  sys.stderr.write("Error 18: The input file contains BOM at the beginning.\n")
  tsv.close()
  sys.exit(18)
tsv.close()

tsv = codecs.open(sys.argv[1], encoding = 'utf-8')

line_counter = 1
for line in tsv:

# check carriage return (CR)
  if line[-2:-1] == "\r":
    sys.stderr.write("Error 24: Input file contains CR characters.\n")
    tsv.close()
    sys.exit(24)

  fields = line[:-1].split("\t")

# check the number of columns
  if len(fields) != number_of_fields:
    sys.stderr.write('Error 14: Check the number of fields in line ' + \
                     str(line_counter) + ".\n")
    tsv.close()
    sys.exit(14)

# check the image file name
  file_name = '^a0[0-9a-f]{6}_[1-9].jpeg$'
  fm = re.fullmatch(file_name, fields[0])
  if fm == None:
    file_name = '^FT[0-9]{3}_[1-9].jpeg$'
    fm = re.fullmatch(file_name, fields[0])
    if fm == None:
      sys.stderr.write('Error 16: The input file does not start with a file name (' + \
                       str(line_counter) + ").\n")
      tsv.close()
      sys.exit(16)

# check the four numbers, coordinates and sizes
  coordinate = '^[1-9][0-9]*$'
  for i in range(1, 5):
    fm = re.fullmatch(coordinate, fields[i])
    if fm == None:
      sys.stderr.write('Error 20: Coordinate or size, ' + str(fields[i]) + ', contains ' + \
                       'unexpected character (' + str(line_counter) + ").\n")
      tsv.close()
      sys.exit(20)

# check the last column and carriage return in Windows text file
  if fields[5] == '0' or fields[5] == '1' or fields[5] == '9':
    pass
  else:
    sys.stderr.write('Error 22: The last column, ' + str(fields[5]) + \
                     ', contains unexpected character (' + str(line_counter) + ").\n")
    tsv.close()
    sys.exit(22)

  line_counter += 1

tsv.close()
