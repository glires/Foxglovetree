#!/usr/bin/env python

# file_files.py - execute command file for all collected files

import os
import glob
import subprocess

volunteers = glob.glob('a000[01]*')	# list of agree ID

for volunteer in volunteers:
  os.chdir(volunteer)
  files = glob.glob('a000[01]*')	# list of collected files
  for file in files:
    subprocess.run(['file', file])	# execute command file for a file
  os.chdir('../')
