#!/usr/bin/env python

'''
prepare_data_for_cnn_loocv.py
prepare data for leave-one-out cross-validation of the 2429 images
'''

__version__ = 0.1
__date__    = '2020-04-25'
__author__  = 'Kohji'

__version__ = 0.2
__date__    = '2020-04-27'

__version__ = 0.3
__date__    = '2020-04-28'

file_images  = './head_shot_data.npy'
file_labels  = './head_shot_labels.pkl'
gpfs         = '/net/gpfs/home/kohji/Okamura_FGT/cross_validation/'
release_size = 359
iv3_size     = 299
margin       = 30	# pixel sized of margin
augmentation = 10	# times of augmentation
flag_ag      = 0	# save augmented images


def read_labels():
  '''
  open and read the label pickle data
  '''
  import pandas
  import pickle
  global head_shot_labels
  with open(file_labels, 'rb') as pickle_file:
    head_shot_labels = pickle.load(pickle_file)


def read_data():
  '''
  open and read the image NumPy data
  '''
  import numpy
  global head_shot_data
  head_shot_data = numpy.load(file_images)


def one_hot_gender_label(is_female):
  '''
  return one-hot gender label in Numpy array
  '''
  import numpy
  gender = numpy.empty((1, 2), dtype = 'int8')
  if is_female:
    gender[0, 0] = 0
    gender[0, 1] = 1
  else:
    gender[0, 0] = 1
    gender[0, 1] = 0
  return gender


def main():
  '''
  the main function
  '''
  import sys
  import cv2
  import numpy
  import keras
  import pandas

  global head_shot_data

  if keras.backend.image_data_format() != 'channels_last':
    sys.stderr.write('Error 20: not channels_last' + "\n")
    sys.exit(20)

  read_labels()
  read_data()

  n_images = head_shot_data.shape[0]
  names = head_shot_labels.index
  if n_images != len(names):
    sys.stderr.write('Error 21: ' + str(n_images) + ' ' + str(len(names)) + "\n")
    sys.exit(21)

  volunteers = {}
  squares = numpy.empty((n_images, iv3_size, iv3_size, 3), dtype = 'float16')
  crop3c  = numpy.empty(          (iv3_size, iv3_size, 3), dtype = 'float16')

  for i, name in enumerate(names):
    if head_shot_labels.iat[i, 0] in volunteers:
      volunteers[head_shot_labels.iat[i, 0]] += "\t" + str(i)
    else:
      volunteers[head_shot_labels.iat[i, 0]] = str(i)

    squared = numpy.reshape(head_shot_data[i], (-1, release_size))
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
    head_shot_data = numpy.empty(((n_images - n_test_images) * augmentation,
                                   iv3_size, iv3_size, 3), dtype = 'float16')
    test_data = numpy.empty((n_test_images, iv3_size, iv3_size, 3), dtype = 'float16')
    labels_train = numpy.empty(((n_images - n_test_images) * augmentation, 2), dtype = 'int8')
    labels_test = numpy.empty((n_test_images, 2), dtype = 'int8')
    counter_train, counter_test = 0, 0
    for i in range(n_images):
      if head_shot_labels.iat[i, 0] == volunteer:
        test_data[counter_test] = squares[i]
        labels_test[counter_test] = one_hot_gender_label(head_shot_labels.iat[i, 2])
        counter_test += 1
      else:
        counter_ag = 0
        for ag in augment.flow(squares[i].reshape((1, iv3_size, iv3_size, 3)), batch_size = 1):
          head_shot_data[counter_train] = ag[0]
          labels_train[counter_train] = one_hot_gender_label(head_shot_labels.iat[i, 2])
          counter_train += 1
          counter_ag += 1
          if flag_ag == 1:
            file_name = 'augmented/ag_' + head_shot_labels.index[i] + \
                '_{:0=3}'.format(counter_ag) + '.jpeg'
            keras.preprocessing.image.save_img(file_name,
                keras.preprocessing.image.array_to_img(head_shot_data[counter_train - 1]))
          if counter_ag >= augmentation:
            break
    numpy.save(gpfs + volunteer + '_tran_data.npy', head_shot_data)
    numpy.save(gpfs + volunteer + '_test_data.npy', test_data)
    numpy.save(gpfs + volunteer + '_tran_labl.npy', labels_train)
    numpy.save(gpfs + volunteer + '_test_labl.npy', labels_test)


if __name__ == '__main__':
  main()
