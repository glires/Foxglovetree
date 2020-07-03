#!/usr/bin/env python

'''
pca.py
perform PCA
'''

__version__ = 0.1
__date__ = '2020-06-27'
__author__ = 'Kohji'

n_volunteers = 277
n_features   =  75


def main():
  '''
  the main function
  '''
  import numpy
  import matplotlib.pyplot
  from pandas import DataFrame
  import sklearn
  from sklearn.decomposition import PCA

  mdb = numpy.load('mandible_vectors.npy')
  data = numpy.empty((n_volunteers, n_features), dtype = float)
  labl = numpy.empty((n_volunteers,), dtype = int)

  data = mdb[:, 3:].astype(float)
  labl = mdb[:, 2].astype(int)
  a = DataFrame(data)

  pca = PCA()
  pca.fit(a)
  feature = pca.transform(a)

  matplotlib.pyplot.scatter(feature[:, 0], feature[:, 1], c = list(labl), cmap = 'bwr')
  matplotlib.pyplot.xlabel('PC1')
  matplotlib.pyplot.ylabel('PC2')
  matplotlib.pyplot.colorbar()
  matplotlib.pyplot.show()


if __name__ == '__main__':
  main()
