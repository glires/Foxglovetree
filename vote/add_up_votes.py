#!/usr/bin/env python

# add_up_votes.py - prepare sex label for all volunteers
# coded by Kohji

import sys
import glob
import numpy

file_volunteers = '/opt/foxglovetree/trim/volunteer_numbers.tsv'
votes = glob.glob('/opt/foxglovetree/votes/20??-*.txt')
members = {'Aki': 0, 'Fuyuki': 1, 'Hiroko': 2, 'Hitomi': 3, 'Kazuaki': 4,
           'Keiko': 5, 'Keisuke': 6, 'Kohji': 7, 'Kosuke': 8, 'Maki': 9,
           'Masao': 10, 'Mayumi': 11, 'Mika': 12, 'Motoko': 13, 'Saki': 14,
           'Sakin': 15, 'Shie': 16, 'Tomoko': 17, 'Yoshikazu': 18,
           'Yoshiko': 19, 'Yuka': 20}
list_volunteers = []
list_agree_id = []
dict_agree_id = {}

n_volunteers = 0
with open(file_volunteers) as file_volunteer:
  for line in file_volunteer:
    fields = line[:-2].split("\t")
    if fields[1] == 'a000ae2f':
      continue
    if fields[1] == 'a000be37':
      continue
    if fields[1] == 'a000cfbd':
      continue
    n_volunteers += 1

voting = numpy.zeros((n_volunteers, len(members)), dtype = int)

i = 0
with open(file_volunteers) as file_volunteer:
  for line in file_volunteer:
    fields = line[:-1].split("\t")
    if fields[1] == 'a000ae2f':
      continue
    if fields[1] == 'a000be37':
      continue
    if fields[1] == 'a000cfbd':
      continue
    list_volunteers.append(fields[0])
    list_agree_id.append(fields[1])
    dict_agree_id[fields[1]] = i
    i += 1

for vote in votes:
  with open(vote) as submission:
    for line in submission:
      fields = line[:-1].split("\t")
      if len(fields) == 3:
        member = fields[0]
        continue
      elif fields[0][0:4] == 'a000':
        if fields[0] == 'a000ae2f':
          continue
        if fields[0] == 'a000be37':
          continue
        if fields[0] == 'a000cfbd':
          continue
        if fields[1] == 'male':
          voting[dict_agree_id[fields[0]]][members[member]] = 7
        elif fields[1] == 'Female':
          voting[dict_agree_id[fields[0]]][members[member]] = 3
        else:
          sys.stderr.write('Error 82: ' + line + "\t" + member + "\n")
          sys.exit(82)
      else:
        sys.stderr.write('Error 84: ' + line + "\t" + member + "\n")
        sys.exit(84)

i = 0
for agree_id in list_agree_id:
  print(list_volunteers[i], end = "\t")
  print(agree_id, end = "\t")
  j = 0
  male = 0
  female = 0
  for member in members:
    if voting[i][j] == 7:
      male += 1
    elif voting[i][j] == 3:
      female += 1
    elif voting[i][j] == 0:
      pass
    else:
      sys.stderr.write('Error 86: ' + agree_id + ' ' + str(i) + ' ' \
                       + str(j) + ' ' + str(voting[i][j]) + "\n")
      sys.exit(86)
    j += 1
  print(male, end = "\t")
  print(female, end = "\t")
  if male > female:
    print('male')
  elif male < female:
    print('female')
  else:	# if equal
      sys.stderr.write('Error 88: ' + agree_id + ' ' + str(male) + ' ' \
                       + str(female) + "\n")
      sys.exit(88)
  i += 1
