#!/usr/bin/env python

'''
learning_curves.py
perform training to draw Figure 5
'''

__version__ = 0.1
__date__    = '2020-07-01'
__author__  = 'Kohji'

n_epochs     = 100
n_classes    = 1	# binary_crossentropy, but not categorical_crossentropy
botrain      = 225	# border of trainable layers
layerl3      = 512
layerl2      = 29
layerl1      = 5
b_size       = 64	# batch_size


def prepare_data_and_labels():
  '''
  read NumPy object files to prepare data and lables
  '''

  import sys
  import numpy

  global data_tran, data_test, labl_tran, labl_test
  data_tran = numpy.load('data_tran.npy')
  data_test = numpy.load('data_test.npy')
  labl_tran = numpy.load('labl_tran.npy')
  labl_test = numpy.load('labl_test.npy')
  if n_classes == 1:	# male: 0.0; female 1.0
    labl_tran = labl_tran[:, 1]
    labl_test = labl_test[:, 1]


def main():
  '''
  the main function
  '''

  import sys
  import numpy
  import keras

  global data_tran, data_test, labl_tran, labl_test
  prepare_data_and_labels()

  if keras.backend.image_data_format() != 'channels_last':
    sys.stderr.write('Error 21: not channels_last' + "\n")
    sys.exit(21)
  base_model = keras.applications.inception_v3.InceptionV3(include_top = False,
                                                           weights = 'imagenet')
  output = base_model.output
  output = keras.layers.pooling.GlobalAveragePooling2D()(output)
  output = keras.layers.core.Dense(units = layerl3, activation = 'relu')(output)
  output = keras.layers.core.Dense(units = layerl2, activation = 'relu')(output)
  output = keras.layers.core.Dense(units = layerl1, activation = 'relu')(output)
  predictions = keras.layers.core.Dense(units = n_classes, activation = 'sigmoid')(output)
  model = keras.models.Model(inputs = base_model.input, outputs = predictions)

  for layer in model.layers[:botrain]:
    layer.trainable = False
    if layer.name.startswith('batch_normalization'):
      layer.trainable = True
  for layer in model.layers[botrain:]:
    layer.trainable = True
  model.compile(optimizer = keras.optimizers.Adam(),
                loss = 'binary_crossentropy', metrics = ['accuracy'])
  checkpoint = keras.callbacks.ModelCheckpoint(filepath = "model_{epoch:02d}.h5",
                                               period = 1)
  model.fit(data_tran, labl_tran, epochs = n_epochs, validation_split = 0,
            batch_size = b_size, callbacks = [checkpoint])

  evaluated = model.evaluate(data_test, labl_test)
  predicted = model.predict(data_test)
  print('loss:', '{:.5f}'.format(evaluated[0]))
  print('accuracy:', '{:.5f}'.format(evaluated[1]))
  print(predicted)


if __name__ == '__main__':
  main()
