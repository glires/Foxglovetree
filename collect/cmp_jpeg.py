#!/usr/bin/env python

# cmp_jpeg.py - compare files to know whether identical files exit
#
# Each volunteer uploads 9 JPEG files. Each of them are supposed to be
# different files. However, he or she sometimes uploads an identical
# file twice or more because 9 image files are too many to handle.
# We have to exclude those identical files for further analysis.

# contents of directories (JPEG files of volunteers)
#
# |-- a00b16f8
# |   |-- a00b16f8.tsv
# |   |-- a00b16f8_1.jpeg
# |   |-- a00b16f8_2.jpeg
# |   |-- a00b16f8_3.jpeg
# |   |-- a00b16f8_4.jpeg
# |   |-- a00b16f8_5.jpeg
# |   |-- a00b16f8_6.jpeg
# |   |-- a00b16f8_7.jpeg
# |   |-- a00b16f8_8.jpeg
# |   `-- a00b16f8_9.jpeg
# |-- a00e1c89
# |   |-- a00e1c89.tsv
# |   |-- a00e1c89_1.jpeg
# |   |-- a00e1c89_2.jpeg
# |   |-- a00e1c89_3.jpeg
# |   |-- a00e1c89_4.jpeg
# |   |-- a00e1c89_5.jpeg
# |   |-- a00e1c89_6.jpeg
# |   |-- a00e1c89_7.jpeg
# |   |-- a00e1c89_8.jpeg
# |   `-- a00e1c89_9.jpeg


import os
import sys
import glob
import subprocess

n_photos = 9	# number of JPEG per volunteer
jpegs = []	# JPEG files for a volunteer
volunteers = glob.glob('a0*')	# list of volunteer IDs
counter = 0	# number of execution of cmp


def compare(starting):
  global jpegs
  global counter
  if os.path.getsize(jpegs[starting]) == 0:
    return
  for i in range(starting + 1, n_photos):
    result = subprocess.run(['cmp', jpegs[starting], jpegs[i]], stdout = subprocess.PIPE)
    counter += 1
    if result.stdout.decode('utf-8') == '':
      print(jpegs[starting], jpegs[i], sep = "\t")
    else:
      pass


for volunteer in volunteers:
  os.chdir(volunteer)
  jpegs = sorted(glob.glob('a0*_?.jpeg'))	# list of JPEG files in a volunteer
  for j in range(n_photos - 1):
    compare(j)
  os.chdir('../')

sys.stderr.write("Number of cmp: " + str(counter) + "\n")
