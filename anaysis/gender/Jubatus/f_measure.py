#!/usr/bin/env python
import sys
tp, fn, fp, tn = 0, 0, 0, 0
for line in sys.stdin:
  fields = line.split("\t")
  if fields[2] == 'tp':
    tp += 1
  elif fields[2] == 'fp':
    fp += 1
  elif fields[2] == 'fn':
    fn += 1
  elif fields[2] == 'tn':
    tn += 1
  else:
    sys.stderr.write('Error 20: ' + line)
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
