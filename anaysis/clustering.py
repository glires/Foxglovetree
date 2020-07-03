#!/usr/bin/env python

'''
clustering.py
perform clustering
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
  from mpl_toolkits.axes_grid1 import make_axes_locatable
  from scipy.spatial.distance  import pdist
  from scipy.cluster.hierarchy import linkage, dendrogram

  mdb = numpy.load('mandible_vectors.npy')
  data = numpy.empty((n_volunteers, n_features), dtype = float)
  labl = numpy.empty((n_volunteers,), dtype = int)

  data = mdb[:, 3:].astype(float)
  labl = mdb[:, 2].astype(int)
  gender = ['male'] * n_volunteers
  for v in range(n_volunteers):
    if labl[v] == 1:
      gender[v] = '{:0=3}'.format(v) + 'female_****'
    else:
      gender[v] = '{:0=3}'.format(v) + 'male'
  a = DataFrame(data)
  a.index = gender

  metric = 'euclidean'
  method = 'average'

  main_axes = matplotlib.pyplot.gca()
  divider = make_axes_locatable(main_axes)

  matplotlib.pyplot.sca(divider.append_axes("left", 1.0, pad = 0))
  ylinkage = linkage(pdist(a, metric = metric), method = method, metric = metric)
  ydendro = dendrogram(ylinkage, orientation = 'left', no_labels = True,
                       distance_sort = 'descending', link_color_func = lambda x: 'black')
  matplotlib.pyplot.gca().set_axis_off()
  a = a.ix[[a.index[i] for i in ydendro['leaves']]]

  matplotlib.pyplot.sca(main_axes)
  matplotlib.pyplot.imshow(a, aspect = 'auto', interpolation = 'none',
                           vmin = -1.0, vmax = 1.0)
  matplotlib.pyplot.colorbar(pad = 0.15)
  matplotlib.pyplot.gca().yaxis.tick_right()
  matplotlib.pyplot.xticks(range(a.shape[1]), a.columns, rotation = 15, size = 'small')
  matplotlib.pyplot.yticks(range(a.shape[0]), a.index, size='small')
  matplotlib.pyplot.gca().xaxis.set_ticks_position('none')
  matplotlib.pyplot.gca().yaxis.set_ticks_position('none')
  matplotlib.pyplot.gca().invert_yaxis()
  matplotlib.pyplot.xticks(fontsize = 6)
  matplotlib.pyplot.yticks(fontsize = 2)
  matplotlib.pyplot.show()


if __name__ == '__main__':
  main()
