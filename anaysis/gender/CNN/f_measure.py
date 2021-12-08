#!/usr/bin/env python

# F-measure of female CNN

import re
import sys
import numpy

file_path = '/net/gpfs/home/kohji/Okamura_FGT/cross_validation/FT'

with open('/mnt/usb3/Foxglovetree/results/results_20200503.txt') as res:
  results = res.readlines()

tp, fn, fp, tn = 0, 0, 0, 0
cnt = 0
for v in range(1, 278):
  labels = numpy.load(file_path + '{:0=3}'.format(v) + '_test_labl.npy')
  if results[cnt][:13] == '## leaving FT':
    vol = int(results[cnt][13:16])
  else:
    sys.stderr.write('Error 22: ' + results[cnt])
    sys.exit(22)
  if v != vol:
    sys.stderr.write('Error 24: ' + 'v = ' + str(v) + ', vol = ' + str(vol) + "\n")
    sys.exit(24)
  cnt += 3
  sigmoids = numpy.empty((9, 1), dtype = int)
  headshot = 0
  while (True):
    if cnt == 3260:	# reading out
      break
    sigmoid = results[cnt]
    re_search = re.search(r'\[+(.+?)\]', sigmoid)
    if re_search:
      sigmoid = float(re_search.group(1))
      sigmoids[headshot, 0] = int(sigmoid + 0.5)
      headshot += 1
      cnt += 1
    else:
      break

  for i in range(headshot):
    if sigmoids[i] == labels[i, 1]:	# if sigmoids[i] == labels[i, 0]:
      if labels[i, 0] == 1:
        tp += 1
      else:
        tn += 1
    else:
      if labels[i, 0] == 1:
        fn += 1
      else:
        fp += 1
sys.stderr.write('tp = ' + str(tp) + "\t")
sys.stderr.write('fn = ' + str(fn) + "\n")
sys.stderr.write('fp = ' + str(fp) + "\t")
sys.stderr.write('tn = ' + str(tn) + "\n")
accuracy  = float(tp + tn) / (tp + fn + fp + tn)
precision = float(tp) / (tp + fp)
recall    = float(tp) / (tp + fn)
f_measure = 2.0 * precision * recall / (precision + recall)
sys.stderr.write('accuracy  = ' + '{:.4f}'.format(accuracy)  + "\n")
sys.stderr.write('precision = ' + '{:.4f}'.format(precision) + "\n")
sys.stderr.write('recall    = ' + '{:.4f}'.format(recall)    + "\n")
sys.stderr.write('F-measure = ' + '{:.4f}'.format(f_measure) + "\n")
