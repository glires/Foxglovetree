#!/usr/bin/env python

'''
jubatus_loocv.py
execute leave-one-out cross-validation
'''

__version__ = 2.2
__date__    = '2020-06-26'
__author__  = 'Kohji'

host = '127.0.0.1'
n_epoch = 100
n_features = 75
exclude = ''	# one will be left out


def train_and_predict(client, data_input):
  '''
  train and precdict
  '''
  import numpy
  import jubatus

  global exclude
  mandible = numpy.load(data_input)
  samples = numpy.arange(mandible.shape[0])	# number of samples

  for epoch in range(n_epoch):
    numpy.random.shuffle(samples)
    numpy.random.shuffle(samples)
    numpy.random.shuffle(samples)	# shuffle sample numbers
    for sample in samples:
      if mandible[sample][0] == int(exclude):
        predict = {}	# only once
        test_label = 'female' if mandible[sample][2] == 1 else 'male'
        for i in range(3, n_features + 3):
          header = 'p{:0=2}'.format(i - 3)
          predict[header] = float(mandible[sample][i])	# 'float16' to float
        predict_data = (jubatus.common.Datum(predict),)	# tuple
      else:
        train = {}
        label = 'female' if mandible[sample][2] == 1 else 'male'
        for i in range(3, n_features + 3):
          header = 'p{:0=2}'.format(i - 3)
          train[header] = float(mandible[sample][i])	# 'float16' to float
        train_data = ((label, jubatus.common.Datum(train)),)	# tuple, tuple
        client.train(train_data)

  result = client.classify(predict_data)	# only one prediction
  predicted = max(result[0], key = lambda x: x.score).label
  print(exclude, end = "\t")
  if test_label == predicted:
    print('correct', end = "\t")
  else:
    print('wrong', end = "\t")
  print(test_label, predicted, result[0], sep = "\t")


def main():
  '''
  the main function
  '''
  import sys
  import jubatus

  global host, n_epoch, exclude
  try:
    port       = int(sys.argv[1])
    data_input =     sys.argv[2]
    exclude    =     sys.argv[3]
    n_epoch    = int(sys.argv[4])
  except:
    sys.stderr.write('Usage: ' + sys.argv[0] +
        ' port_number input_data.npy excluding_id number_of_epoch' + "\n")
    sys.exit(70)

  client = jubatus.classifier.client.Classifier(host, port, 'two-class')
  train_and_predict(client, data_input)


if __name__ == '__main__':
  main()
