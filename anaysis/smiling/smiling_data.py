#!/usr/bin/env python

'''
smiling_data.py
prepare training data, training labels, test data, and test labels for smiling analysis
'''

__author__  = 'Kohji'
__version__ = 0.1
__date__    = '2021-06-10'
__version__ = 0.2
__date__    = '2021-06-22'

path_data    = '/home/kohji/Foxglovetree/'

file_images  = path_data + 'data/headshot_data.npy'
file_labels  = path_data + 'data/headshot_labels.npy'
gpfs         = path_data + 'cross_validation/'
size_release = 359
size_iv3     = 299
margin       = 30	# in pixel
fold_cro_val = 10
augmentation = 10	# times of augmentation
threshold    = 0.8


def read_labels():
  '''
  read headshot labels
  '''
  import numpy

  global headshot_labels, flags_smiling, n_smiling
  headshot_labels = numpy.load(file_labels)
  n = len(headshot_labels)
  flags_smiling = numpy.zeros((n,), dtype = 'int8')
  n_smiling = 0
  for headshot in range(n):
    nonsm = headshot_labels[headshot][3]
    smile = headshot_labels[headshot][4]
    if smile / float(nonsm + smile) > threshold:
      flags_smiling[headshot] = 2
      n_smiling += 1	# n_smiling will be 668 in the end
    elif smile == 0:
      flags_smiling[headshot] = 1

def read_data():
  '''
  read headshot data
  '''
  import numpy
  global headshot_data
  headshot_data = numpy.load(file_images)

def main():
  '''
  prepare data and labels as NumPy file
  '''
  import sys
  import cv2
  import numpy
  import keras

  if keras.backend.image_data_format() != 'channels_last':
    error = 20
    sys.stderr.write('Error ' + str(error) + ': not channels_last' + "\n")
    sys.exit(error)

  global headshot_data, headhot_labels, flags_smiling, n_smiling
  read_labels()
  read_data()

  n_images = len(headshot_data)

  squaren = numpy.empty((n_smiling, size_iv3, size_iv3, 3), dtype = 'float16')
  squares = numpy.empty((n_smiling, size_iv3, size_iv3, 3), dtype = 'float16')
  crop3c  = numpy.empty(     (size_iv3, size_iv3, 3), dtype = 'float16')

  i = 0	# to get fist n_smiling of non-smiling in order to maintain balance
  for headshot in range(n_images):
    if flags_smiling[headshot] == 1:
      squared = numpy.reshape(headshot_data[headshot], (-1, size_release))
      crop = squared[margin:(margin + size_iv3), margin:(margin + size_iv3)]
      for channel in range(3):
        crop3c[:size_iv3, :size_iv3, channel] = crop
      squaren[i] = crop3c / 255.0
      i += 1
      if i >= n_smiling:	# only the first n_smiling headshots
        break

  i = 0	# smiling
  for headshot in range(n_images):
    if flags_smiling[headshot] == 2:
      squared = numpy.reshape(headshot_data[headshot], (-1, size_release))
      crop = squared[margin:(margin + size_iv3), margin:(margin + size_iv3)]
      for channel in range(3):
        crop3c[:size_iv3, :size_iv3, channel] = crop
      squares[i] = crop3c / 255.0
      i += 1

  augment = keras.preprocessing.image.ImageDataGenerator(rotation_range = 6.0,
      width_shift_range = 0.06, height_shift_range = 0.06,
      shear_range = 2, zoom_range = 0.08, fill_mode = 'nearest',
      horizontal_flip = True, vertical_flip = False,
      data_format = 'channels_last', validation_split = 0.0)

  shuffled = numpy.arange(n_smiling)
  numpy.random.shuffle(shuffled)	# just prepare shuffled list
  numpy.random.shuffle(shuffled)
  numpy.random.shuffle(shuffled)

  n_test = n_smiling // fold_cro_val	# 10-fold cross-validation; n_test may be 66
  for batch in range(0, n_smiling, n_test):
    set = batch // n_test
    if set >= fold_cro_val:	# ignore last remainders
      break
    tran_data = numpy.empty(((n_smiling - n_test) * 2 * augmentation,
                    size_iv3, size_iv3, 3), dtype = 'float16')
    test_data = numpy.empty((n_test * 2, size_iv3, size_iv3, 3), dtype = 'float16')
    tran_labl = numpy.empty(((n_smiling - n_test) * 2 * augmentation, 2), dtype = 'int8')
    test_labl = numpy.empty((n_test * 2, 2), dtype = 'int8')
    cnt_tran, cnt_test = 0, 0
    for i in range(n_smiling):
      if batch <= i < (batch + n_test):
        test_data[cnt_test] = squaren[shuffled[i]]
        test_labl[cnt_test] = numpy.array([1, 0], dtype = 'int8')
        cnt_test += 1
        test_data[cnt_test] = squares[shuffled[i]]
        test_labl[cnt_test] = numpy.array([0, 1], dtype = 'int8')
        cnt_test += 1
      else:
        cnt_ag = 0
        for ag in augment.flow(squaren[shuffled[i]].reshape((1, size_iv3, size_iv3, 3)), batch_size = 1):
          tran_data[cnt_tran] = ag[0]
          tran_labl[cnt_tran] = numpy.array([1, 0], dtype = 'int8')
          cnt_tran += 1
          cnt_ag += 1
          if cnt_ag >= augmentation:
            break
        for ag in augment.flow(squares[shuffled[i]].reshape((1, size_iv3, size_iv3, 3)), batch_size = 1):
          tran_data[cnt_tran] = ag[0]
          tran_labl[cnt_tran] = numpy.array([0, 1], dtype = 'int8')
          cnt_tran += 1
          cnt_ag += 1
          if cnt_ag >= (augmentation * 2):
            break
    numpy.save(gpfs + 'smile' + str(set) + '_tran_data.npy', tran_data)
    numpy.save(gpfs + 'smile' + str(set) + '_test_data.npy', test_data)
    numpy.save(gpfs + 'smile' + str(set) + '_tran_labl.npy', tran_labl)
    numpy.save(gpfs + 'smile' + str(set) + '_test_labl.npy', test_labl)

if __name__ == '__main__':
  main()
