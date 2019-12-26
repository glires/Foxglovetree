#!/usr/bin/env python

# Usage: juba_train_and_predict.py -h

import sys
import json
import numpy
import random
from jubatus.classifier.client import Classifier
from jubatus.common import Datum


number_of_epoch = 50


def _output(unicode_value):
  stdout = sys.stdout.buffer
  stdout.write(unicode_value.encode('utf-8'))


def train_and_predict(client, file):
  input_data = []
  number_of_samples = 0
  with open(file) as tsv:
    line = tsv.readline()
    header = line[:-1].split("\t")
    for line in tsv:
      if line[8:11] == 'CHB':
        input_data.append(line)
        number_of_samples += 1
      elif line[8:11] == 'JPT':
        input_data.append(line)
        number_of_samples += 1
      else:
        continue
  shuffled_numbers = numpy.arange(number_of_samples)

  for epoch in range(number_of_epoch):
    random.shuffle(shuffled_numbers)
    random.shuffle(shuffled_numbers)
    random.shuffle(shuffled_numbers)
    for i in shuffled_numbers:
      fields = input_data[i][:-1].split("\t")
      if fields[0][0:7] == exclude:
        predict_data = []
        predict = {}
        answer = fields[0][8:11]
        for j in range(1, len(fields)):
          fields[j] = float(fields[j])
          predict.update({header[j]: fields[j]})
        predict_data.append((Datum(predict)))
      else:
        train_data = []
        trains = {}
        for j in range(1, len(fields)):
          fields[j] = float(fields[j])
          trains.update({header[j]: fields[j]})
        train_data.append((fields[0][8:11], Datum(trains)))
        client.train(train_data)

  result = client.classify([predict_data[0]])
  predicted = max(result[0], key = lambda x: x.score).label
  if answer == predicted:
    print('correct', end = "\t")
  else:
    print('wrong', end = "\t")
  print(answer, predicted, result, sep = "\t")


if __name__ == '__main__':

  try:
    exclude = sys.argv[3]
    training = sys.argv[2]
    port = int(sys.argv[1])
  except:
    sys.stderr.write("Usage: jubatus.py port_number training.tsv exclude name\n")
    sys.exit(7)

  localhost = '127.0.0.1'
  if len(sys.argv) > 4:
    name = sys.argv[4]
  else:
    name = 'Coded by Kohji'

  client = Classifier(localhost, port, name)	# connect to Jubatus
  train_and_predict(client, training)
