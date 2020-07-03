#!/usr/bin/env python

'''
lightgbm_loocv.py
perform leave-one-out cross-validation
'''

__version__ = 0.6
__date__    = '2020-06-18'
__author__  = 'Kohji'

__version__ = 0.7
__date__    = '2020-06-21'

import numpy

file_mandible_vectors = 'mandible_vectors.npy'
n_faces = 277
n_elements = 75

mandible_data = numpy.empty((n_faces, n_elements), dtype = 'float16')
mandible_labl = numpy.empty((n_faces,), dtype = 'int8')


def prepare_data_and_labels():
  '''
  prepare data and labels
  '''
  import numpy
  global mandible_data, mandible_labl
  mandible_vectors = numpy.load(file_mandible_vectors)
  mandible_data = mandible_vectors[:, 3:(n_elements + 3)]
  mandible_labl = mandible_vectors[:, 2].astype('int8')


def main():
  '''
  the main function
  '''
  import sys
  import numpy
  import lightgbm

  global mandible_data, mandible_labl
  prepare_data_and_labels()

  for face in range(n_faces):	# selecting one test face
    tran_data = numpy.delete(mandible_data, face, 0)
    tran_labl = numpy.delete(mandible_labl, face)
    test_data = mandible_data[face].reshape(1, n_elements)
    test_labl = mandible_labl[face].reshape(1, 1)

    lgbm_tran = lightgbm.Dataset(tran_data, tran_labl)
    lgbm_test = lightgbm.Dataset(test_data, test_labl, reference = lgbm_tran)

    # lgbm_params = { 'objective': 'multiclass', 'num_class': 2 }
    lgbm_params   = { 'objective': 'binary', 'metric': 'binary_logloss' }
    model = lightgbm.train(lgbm_params, lgbm_tran, num_boost_round = 38)

    pred = model.predict(test_data, num_iteration = model.best_iteration)
    # pred_max = numpy.argmax(pred, axis = 1)	# pred (1, 2) for multiclass
    pred_max = int(pred + 0.5)	# pred (1,) for binary
    print(face, end = "\t")
    if test_labl == pred_max:
      print('correct')
    else:
      print('wrong')


if __name__ == '__main__':
  main()
