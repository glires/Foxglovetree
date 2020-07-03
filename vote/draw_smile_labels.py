#!/usr/bin/env python
'''
draw_smile_labels.py
draw smile labels
'''

npy_labels = 'headshot_labels.npy'
n_voters = 17

def main():
  '''
  the main function
  '''
  import numpy
  import matplotlib.pyplot

  smiling = numpy.load(npy_labels)
  sorted_smiling = sorted(smiling[:, 4])
  n_images = len(sorted_smiling)
  figure = numpy.empty((n_voters, n_images), dtype = int)
  for i in range(n_images):
    non_smile = n_voters - sorted_smiling[i]
    marking = 0
    for s in range(non_smile):
      figure[s, i] = 0
      marking += 1
    while marking < n_voters:
      figure[marking, i] = 1
      marking += 1
  matplotlib.pyplot.imshow(figure, aspect = 'auto', cmap = 'winter')
  matplotlib.pyplot.colorbar()
  matplotlib.pyplot.xlabel('Headshots', fontsize = 14)
  matplotlib.pyplot.ylabel('Votes', fontsize = 14)
  matplotlib.pyplot.show()

if __name__ == '__main__':
  main()
