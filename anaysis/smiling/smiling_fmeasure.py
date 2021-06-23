#!/usr/bin/env python

'''
smiling_fmeasure.py
calculate F-measure of the smiling recognition CNN
'''

__author__ = 'Kohji'
__version__ = 0.1
__date__    = '2021-06-13'
__version__ = 0.2
__date__    = '2021-06-14'
__version__ = 0.3
__date__    = '2021-06-23'

def main():
  import re
  import sys
  import numpy

  with open('predictions.txt') as pred:
    prediction = pred.readlines()

  tp, fn, fp, tn = 0, 0, 0, 0
  cnt = 0
  for i in range(10):
      labels = numpy.load('smile' + str(i) + '_test_labl.npy')
      print(prediction[135 * i + 2], end = '')	# accuracy
      for j in range(66 * 2):	# number of test samples
        sigmoid = prediction[135 * i + 3 + j]
        re_search = re.search(r'\[+(.+?)\]+', sigmoid)
        if re_search:
          sigmoid = float(re_search.group(1))
          smiling = int(sigmoid + 0.5)
        else:
          error = 20
          sys.stderr.write('Error ' + str(error) + ': ' + sigmoid + "\n")
          sys.exit(error)
        if smiling == labels[j, 1]:
          if labels[j, 1] == 1:
            tp += 1
          else:
            tn += 1
        else:
          if labels[j, 1] == 1:
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

if __name__ == '__main__':
  main()
