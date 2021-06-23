#!/usr/bin/env python

'''
smiling_validation.py
perform 10-fold cross-validation for the smiling headshots
'''

__author__  = 'Kohji'
__version__ = 0.6
__date__    = '2021-06-10'

n_epochs     = 20
n_classes    = 1	# binary_crossentropy, but not categorical_crossentropy
botrain      = 225	# border of trainable layers
layerl3      = 512
layerl2      = 29
layerl1      = 5
v_split      = 0	# validation_split
b_size       = 64	# batch_size
predicts     = 'predictions.txt'

def prepare_data_and_labels():
  import sys
  import numpy
  global tran_data, tran_labl, test_data, test_labl
  tran_data = numpy.load(sys.argv[1] + '_tran_data.npy')
  tran_labl = numpy.load(sys.argv[1] + '_tran_labl.npy')
  test_data = numpy.load(sys.argv[1] + '_test_data.npy')
  test_labl = numpy.load(sys.argv[1] + '_test_labl.npy')
  if n_classes == 1:
    tran_labl = tran_labl[:, 1]
    test_labl = test_labl[:, 1]

def main():
  import sys
  import numpy
  import keras

  if len(sys.argv) != 2:
    sys.stderr.write('Error 20: no volunteer ID provided' + "\n")
    sys.exit(20)
  global tran_data, tran_labl, test_data, test_labl
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
  # print(model.summary())

  model.fit(tran_data, tran_labl, epochs = n_epochs,
            validation_split = v_split, batch_size = b_size)
  # model.save('trained_' + sys.argv[1] + '.h5', include_optimizer = False)

  evaluated = model.evaluate(test_data, test_labl)
  predicted = model.predict(test_data)
  with open(predicts, 'a') as pred:
    print('## dataset', sys.argv[1], predicted.shape, file = pred)
    print('loss:', '{:.5f}'.format(evaluated[0]), file = pred)
    print('accuracy:', '{:.5f}'.format(evaluated[1]), file = pred)
    print(predicted, file = pred)

if __name__ == '__main__':
  main()
