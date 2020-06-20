#!/usr/bin/env python

'''
mandible_vector.py
prepare mandible vectors from the 359x359 headshot data

This script makes a simple dataset for data science
by reading the following two headshot NumPy files.
The first and second columns retain volunteer numbers from 1 to 277
and photo numbers from 1 to 9.
The third one retains female labels, namely 0 for male or 1 for female.
Following 75 columes retain 75 features or explanatory variables
that derived from horizontal pixcel data around jaw of each volunteer.
The more column number increases, the more inside, around mouth,
feature represents.
The values have been normalized from 0.0 to 1.0.
'''

__version__ = 0.1
__date__    = '2020-05-30'
__author__  = 'Kohji'

__version__ = 0.2
__date__    = '2020-06-18'

__version__ = 0.3
__date__    = '2020-06-20'

file_headshot_data    = 'headshot_data.npy'
file_headshot_labl    = 'headshot_labels.npy'
file_mandible_vectors = 'mandible_vectors.npy'
n_volunteer = 277	# number of volunteers
n_elements  = 75	# number of features or explanatory variables
size_square = 359	# pixel size of the input squre image


def print_usage():
  '''
  print usage
  '''
  import sys
  sys.stderr.write('Usage: ' + sys.argv[0] +
      ' [-h] [-o mandible_vectors.npy] [-v]' + "\n")
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

  mandible_vectors = numpy.empty((n_volunteer, n_elements + 3), dtype = 'float16')
	# column 0: volunteer number, 1 to 277
	# column 1: photo number, 1 to 9
	# colunn 2: female lable, 0 (male) or 1 (female)
	# columns 3 to 77: horizontal pixel data, from outside to around mouth

  shuffled = numpy.arange(n_volunteer)
  numpy.random.shuffle(shuffled)	# shuffle the order of volunteers
  sfld = 0	# index for shuffled
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
        mandible_vectors[shuffled[sfld], (j - 30) // 2 + 3] = v / 12.0 / 255.0	# normalized
      v = image[267, 178] + image[267, 179] + image[267, 180] \
        + image[268, 178] + image[268, 179] + image[268, 180] \
        + image[269, 178] + image[269, 179] + image[269, 180]
      mandible_vectors[shuffled[sfld], 0] = volunteer
      mandible_vectors[shuffled[sfld], 1] = headshot_labl[i, 1]	# photo number
      mandible_vectors[shuffled[sfld], 2] = headshot_labl[i, 2]	# female label
      mandible_vectors[shuffled[sfld], n_elements + 2] = v / 9.0 / 255.0	# the center
      sfld += 1
  if sfld != n_volunteer:
    sys.stderr.write('Error 20: sfld = ' + str(sfld) + "\n")
    sys.exit(20)

  numpy.save(file_mandible_vectors, mandible_vectors)


if __name__ == '__main__':
  main()
