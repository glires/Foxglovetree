#!/usr/bin/env python

'''
prepare_training_and_test_data.py
prepare data for leave-one-out cross-validation of the 2429 images
'''

__author__  = 'Kohji'
__version__ = 0.1
__date__    = '2020-04-25'
__version__ = 0.2
__date__    = '2020-04-27'
__version__ = 0.3
__date__    = '2020-04-28'
__version__ = 0.4
__date__    = '2021-12-10'

file_images  = './headshot_data.npy'
file_labels  = './headshot_labels.npy'
dir_data     = './cross_validation/'
release_size = 359
iv3_size     = 299
margin       = 30	# pixel sized of margin
augmentation = 10	# times of augmentation
flag_ag      = 0	# save augmented images

import numpy

def read_data():
  '''
  open and read the image data
  '''
  global headshot_data
  headshot_data = numpy.load(file_images)

def read_labels():
  '''
  open and read the label data
  '''
  global headshot_labels
  headshot_labels = numpy.load(file_labels)

def one_hot_gender_label(is_female):
  '''
  return one-hot gender label in Numpy array
  '''
  if is_female: return numpy.array((0, 1), dtype = 'int8')
  else:         return numpy.array((1, 0), dtype = 'int8')	# male

def main():
  '''
  the main function
  '''
  import sys
  import keras

  global headshot_data, headshot_labels

  if keras.backend.image_data_format() != 'channels_last':
    error = 40
    sys.stderr.write('Error ' + str(error) + ': not channels_last\n')
    sys.exit(error)

  read_labels()
  read_data()

  n_images = headshot_data.shape[0]
  names = headshot_labels[:, 0]
  if n_images != len(names):
    error = 42
    sys.stderr.write('Error ' + str(error) + ': ' + str(n_images) + ' ' + str(len(names)) + '\n')
    sys.exit(error)

  volunteers = {}
  squares = numpy.empty((n_images, iv3_size, iv3_size, 3), dtype = 'float16')
  crop3c  = numpy.empty(          (iv3_size, iv3_size, 3), dtype = 'float16')

  for i, name in enumerate(names):
    if headshot_labels[i, 0] in volunteers:
      volunteers[headshot_labels[i, 0]] += "\t" + str(i)
    else:
      volunteers[headshot_labels[i, 0]] = str(i)

    squared = numpy.reshape(headshot_data[i], (-1, release_size))
    crop = squared[margin:(margin + iv3_size), margin:(margin + iv3_size)]
    for channel in range(3):
      crop3c[:iv3_size, :iv3_size, channel] = crop
    squares[i] = crop3c / 255.0

  augment = keras.preprocessing.image.ImageDataGenerator(rotation_range = 6.0,
      width_shift_range = 0.06, height_shift_range = 0.06,
      shear_range = 2, zoom_range = 0.08, fill_mode = 'nearest',
      horizontal_flip = True, vertical_flip = False,
      data_format = 'channels_last', validation_split = 0.0)

  for volunteer in sorted(volunteers.keys()):
    lines = volunteers[volunteer].split("\t")
    n_test_images = len(lines)
    tran_data = numpy.empty(((n_images - n_test_images) * augmentation,
                    iv3_size, iv3_size, 3), dtype = 'float16')
    test_data = numpy.empty((n_test_images, iv3_size, iv3_size, 3), dtype = 'float16')
    tran_labl = numpy.empty(((n_images - n_test_images) * augmentation, 2), dtype = 'int8')
    test_labl = numpy.empty((n_test_images, 2), dtype = 'int8')

    counter_train, counter_test = 0, 0
    for i in range(n_images):
      if headshot_labels[i, 0] == volunteer:
        test_data[counter_test] = squares[i]
        test_labl[counter_test] = one_hot_gender_label(headshot_labels[i, 2])
        counter_test += 1
      else:
        counter_ag = 0
        for ag in augment.flow(squares[i].reshape((1, iv3_size, iv3_size, 3)), batch_size = 1):
          tran_data[counter_train] = ag[0]
          tran_labl[counter_train] = one_hot_gender_label(headshot_labels[i, 2])
          counter_train += 1
          counter_ag += 1
          if flag_ag == 1:
            file_name = 'augmented/ag_{:0=3}'.format(headshot_labels[i]) + \
                '_{:0=3}'.format(counter_ag) + '.jpeg'
            keras.preprocessing.image.save_img(file_name,
                keras.preprocessing.image.array_to_img(tran_data[counter_train - 1]))
          if counter_ag >= augmentation:
            break

    numpy.save(dir_data + 'lo{:0=3}'.format(volunteer) + '_tran_data.npy', tran_data)
    numpy.save(dir_data + 'lo{:0=3}'.format(volunteer) + '_test_data.npy', test_data)
    numpy.save(dir_data + 'lo{:0=3}'.format(volunteer) + '_tran_labl.npy', tran_labl)
    numpy.save(dir_data + 'lo{:0=3}'.format(volunteer) + '_test_labl.npy', test_labl)

if __name__ == '__main__': main()
