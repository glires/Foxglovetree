#!/usr/bin/env python

'''
randomforest.py
execute randomforest for the 385 samples
'''

__version__ = 2.1
__date__    = '2020-06-25'
__author__  = 'Kohji'


data_input  = 'float_data_20200617.tsv'
samples_385 = '/net/gpfs/home/kohji/Okamura_JUB/data_Nishino_20190510/For_Classification/samples_385.tsv'
n_samples = 385
n_probes  = 413245
n_b_round = 75

import sys
import numpy
sample385_data = numpy.empty((n_samples, n_probes), dtype = 'float16')
sample385_labl = numpy.empty((n_samples,), dtype = 'int8')
sample_names = []


def prepare_data_and_labels():
  '''
  prepare data and labels
  '''

  with open(data_input) as tsv:
    line = tsv.readline()	# read out header
    sample = 0
    for line in tsv:
      fields = line[:-1].split("\t")
      for feature in range(n_probes):
        sample385_data[sample, feature] = fields[feature + 1]
      if fields[0][8:10] == 'sm':
        sample385_labl[sample] = 0
      elif fields[0][8:10] == 'ps':
        sample385_labl[sample] = 1
      else:
        sys.stderr.write('Error 20: unknown label, ' + fields[0] + "\n")
        sys.exit(20)
      sample += 1


def prepare_sample_names():
  '''
  prepare sample names
  '''

  global sample_names
  with open(samples_385) as s385:
    for line in s385:
      fields = line.split("\t")
      sample_names.append(fields[0])


def main():
  '''
  the main function
  '''
  from sklearn.ensemble import RandomForestClassifier

  prepare_data_and_labels()
  prepare_sample_names()

  for sample in range(n_samples):
    tran_data = numpy.delete(sample385_data, sample, 0)
    tran_labl = numpy.delete(sample385_labl, sample)
    test_data = sample385_data[sample].reshape(1, -1)
    test_labl = sample385_labl[sample].reshape(1, -1)

    model = RandomForestClassifier(max_depth = 32, n_estimators = 16, random_state = 32)
    model.fit(tran_data, tran_labl)

    pred = model.predict(test_data)
    print('random_forest', sample_names[sample], sep = "\t", end = "\t")
    if test_labl == pred[0]:
      print('correct')
    else:
      print('wrong')


if __name__ == '__main__':
  main()
