#!/usr/bin/env python

'''
mandible_vector.py
prepare mandible vectors from the 359x359 headshot data

This script makes a simple dataset for data science
by reading the following two NumPy files.
The first column retains volunteer numbers from 1 to 277.
Following 75 columes retain 75 explanatory variables
that derived from horizontal pixcel data around jaw of each volunteer.
The last column retains female label.
'''

__version__ = 0.1
__date__    = '2020-05-30'
__author__  = 'Kohji'

__version__ = 0.2
__date__    = '2020-06-18'

file_headshot_data    = 'head_shot_data.npy'
file_headshot_labl    = 'head_shot_labels.npy'
file_mandible_vectors = 'mandible_vectors.npy'
n_volunteer = 277	# number of volunteers
n_elements  = 75	# number of explanatory variables
size_square = 359	# pixel size of the input squre image


def print_usage():
  '''
  print usage
  '''
  import sys
  sys.stderr.write("Usage: mandible_vector.py [-h] [-o mandible_vectors.npy] [-v]\n")
  sys.exit(1)


def main():
  '''
  the main function
  '''
  import sys
  import numpy
  import getopt

  global file_mandible_vectors
  try:
    opts, args = getopt.getopt(sys.argv[1:], 'ho:v')
  except:
    print_usage()
  for option in opts:
    if option[0] == '-o':
      file_mandible_vectors = option[1]
    elif option[0] == '-v':
      sys.stderr.write('mandible_vector.py (' + sys.argv[0] + ') ver. ' \
                       + str(__version__) + "\n")
      sys.exit(2)
    else:
      print_usage()

  headshot_data = numpy.load(file_headshot_data)
  headshot_labl = numpy.load(file_headshot_labl)
  n_images = headshot_labl.shape[0]

  mandible_vectors = numpy.empty((n_volunteer, n_elements + 2), dtype = 'float16')
	# column 0: volunteer number
	# columns 1 to n_elements: data
	# last column (n_elements + 1): female label (0: male; 1: female)

  volunteer_numbers = set()	# to reduce redundancy
  for i in range(n_images):
    if headshot_labl[i, 0] in volunteer_numbers:
      continue	# one vector per volunteer
    else:
      volunteer = headshot_labl[i, 0]
      volunteer_numbers.add(volunteer)
      image = headshot_data[i].reshape(-1, size_square).astype(int)
      for j in range(30, 178, 2):
        v = image[267, j] + image[267, j + 1] + image[267, 358 - j] + image[267, 357 - j] \
          + image[268, j] + image[268, j + 1] + image[268, 358 - j] + image[268, 357 - j] \
          + image[269, j] + image[269, j + 1] + image[269, 358 - j] + image[269, 357 - j]
        mandible_vectors[volunteer - 1, (j - 30) // 2 + 1] = v / 12.0 / 255.0
      v = image[267, 178] + image[267, 179] + image[267, 180] \
        + image[268, 178] + image[268, 179] + image[268, 180] \
        + image[269, 178] + image[269, 179] + image[269, 180]
      mandible_vectors[volunteer - 1, n_elements] = v / 9.0 / 255.0	# the center
      mandible_vectors[volunteer - 1, n_elements + 1] = headshot_labl[i, 2]	# female label
      mandible_vectors[volunteer - 1, 0] = volunteer

  numpy.save(file_mandible_vectors, mandible_vectors)


if __name__ == '__main__':
  main()
