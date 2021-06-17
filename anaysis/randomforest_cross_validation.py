#!/usr/bin/env python

'''
randomforest.py
execute leave-one-out cross-validation for the Foxglovetree samples
'''
__version__ = 0.1
__date__    = '2021-06-10'
__author__  = 'Kohji'

iv3_size   = 299
n_features = iv3_size ** 2

import sys
import numpy

name = 'FT{:0=3}'.format(int(sys.argv[1]))
tdata = numpy.load('/net/gpfs/home/kohji/Okamura_FGT/cross_validation/' +
            name + '_tran_data.npy')
n_samples = tdata.shape[0]
tran_data = numpy.empty((n_samples, n_features), dtype = 'float16')
tran_labl = numpy.empty((n_samples,), dtype = 'int8')
sample_names = []

def prepare_data_and_labels():
  '''
  prepare data and labels
  '''
  for sample in range(n_samples):
    feature = 0
    for i in range(iv3_size):
      for j in range(iv3_size):
        tran_data[sample, feature] = tdata[sample, i, j, 0]
        feature += 1
  data = numpy.load('/net/gpfs/home/kohji/Okamura_FGT/cross_validation/' +
             name + '_tran_labl.npy')
  for sample in range(n_samples):
    tran_labl[sample] = data[sample, 0]

def prepare_sample_names():
  '''
  prepare sample names
  '''
  global sample_names
  for sample in range(n_samples):
    sample_names.append('{:0=5}'.format(sample))

def main():
  '''
  the main function
  '''
  from sklearn.ensemble import RandomForestClassifier

  prepare_data_and_labels()
  prepare_sample_names()

  model = RandomForestClassifier(max_depth = 32, n_estimators = 16, random_state = 32)
  model.fit(tran_data, tran_labl)

  data = numpy.load('/net/gpfs/home/kohji/Okamura_FGT/cross_validation/' +
             name + '_test_data.npy')
  test_data = numpy.empty((data.shape[0], n_features), dtype = 'float16')
  for sample in range(data.shape[0]):
    feature = 0
    for i in range(iv3_size):
      for j in range(iv3_size):
        test_data[sample, feature] = data[sample, i, j, 0]
        feature += 1
  data = numpy.load('/net/gpfs/home/kohji/Okamura_FGT/cross_validation/' +
             name + '_test_labl.npy')
  test_labl = data[:, 0].reshape(data.shape[0], -1)

  for sample in range(test_data.shape[0]):
    pred = model.predict(test_data[sample].reshape(1, -1))
    print(name, sample, sep = "\t", end = "\t")
    if test_labl[sample] == pred[0]:
      print('correct')
    else:
      print('wrong')

if __name__ == '__main__':
  main()
