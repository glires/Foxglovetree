#!/usr/bin/env python

'''
mandible_randomforest.py
perform random forest classification
'''

__author__  = 'Kohji'
__version__ = 0.1
__date__    = '2020-06-24'
__version__ = 0.2
__date__    = '2021-06-14'

import sys
import numpy

file_mandible_vectors = '../data/mandible_vectors.npy'
n_faces = 277
n_elements = 75

mandible_data = numpy.empty((n_faces, n_elements), dtype = 'float16')
mandible_labl = numpy.empty((n_faces,), dtype = 'int8')

def prepare_data_and_labels():
  global mandible_data, mandible_labl
  mandible_vectors = numpy.load(file_mandible_vectors)
  mandible_data = mandible_vectors[:, 3:(n_elements + 3)]
  mandible_labl = mandible_vectors[:, 2].astype('int8')

def main():
  from sklearn.ensemble import RandomForestClassifier
  global mandible_data, mandible_labl
  prepare_data_and_labels()

  tp, fn, fp, tn = 0, 0, 0, 0

  for face in range(n_faces):	# selecting one test face
    tran_data = numpy.delete(mandible_data, face, 0)
    tran_labl = numpy.delete(mandible_labl, face)
    test_data = mandible_data[face].reshape(1, n_elements)
    test_labl = mandible_labl[face].reshape(1, 1)

    model = RandomForestClassifier(max_depth = 32, n_estimators = 16, random_state = 32)
    model.fit(tran_data, tran_labl)

    pred = model.predict(test_data)
    print(face, end = "\t")
    if test_labl == pred[0]:
      print('correct', end = "\t")
      if test_labl == 1:
        print('tp')
        tp += 1
      else:
        print('tn')
        tn += 1
    else:
      print('wrong', end = "\t")
      if test_labl == 1:
        print('fn')
        fn += 1
      else:
        print('fp')
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
