#!/usr/bin/env python

# cross_validate.py

import sys
import numpy
import keras
import pandas
import random
import getopt
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import SGD

exclude = ''
input_data = 'data.tsv'
output_results = 'validation.tsv'
usage = "Usage (example): cross_validate.py -e NA18541 data.tsv\n"
excluding_row = -1
populations = 4	# YRI, CEU, CHB, and JPT
number_of_batch_size = 64
number_of_epochs = 100

try:
  opts, args = getopt.getopt(sys.argv[1:], 'e:h')
except:
  sys.stderr.write('Option error: ')
  i = 0
  for option in sys.argv:
    if i > 0:
      sys.stderr.write(' ' + option)
    i += 1
  sys.stderr.write("\n")
  sys.stderr.write(usage)
  sys.exit(76)

for option in opts:
  if option[0] == '-e':
    exclude = option[1]
  elif option[0] == '-h':
    sys.stderr.write(usage)
    sys.exit(74)
  else:
    sys.stderr.write(usage)
    sys.exit(72)
if len(args) == 1:
  input_data = args[0]	# over write
elif len(args) == 0:
  pass
else:
  sys.stderr.write(usage)
  sys.exit(78)

data = pandas.read_csv(input_data, sep = "\t", index_col = 0)
number_of_data = len(data)
number_of_columns = data.columns

if exclude == '':
  excluding_row = -2
else:
  i = 0
  for id in data.index:
    if id[0:7] == exclude:
      excluding_row = i
      break
    i += 1
  number_of_data -= 1

if excluding_row == -1:
  sys.stderr.write('ID ' + exclude + ' not found' + "\n")
  sys.exit(80)

shuffled_numbers = numpy.arange(len(data))
random.shuffle(shuffled_numbers)
random.shuffle(shuffled_numbers)
random.shuffle(shuffled_numbers)
random.shuffle(shuffled_numbers)
random.shuffle(shuffled_numbers)

teacher = numpy.zeros((number_of_data, populations), dtype = int)
answer = numpy.zeros((1, populations), dtype = int)
i = 0
for id in shuffled_numbers:
  pop = (data.index)[id][8:11]
  if id == excluding_row:
    if pop == 'YRI':
      answer[0, 0] = 1
    elif pop == 'CEU':
      answer[0, 1] = 1
    elif pop == 'CHB':
      answer[0, 2] = 1
    elif pop == 'JPT':
      answer[0, 3] = 1
    else:
      sys.stderr.write('Error: ' + pop + str(id) + "\n")
      sys.exit(82)
    testing = numpy.array(data.iloc[id, :], dtype = float)
    testing = numpy.reshape(testing, (1, -1))
  else:
    if pop == 'YRI':
      teacher[i, 0] = 1
    elif pop == 'CEU':
      teacher[i, 1] = 1
    elif pop == 'CHB':
      teacher[i, 2] = 1
    elif pop == 'JPT':
      teacher[i, 3] = 1
    else:
      sys.stderr.write('Error: ' + pop + str(id) + "\n")
      sys.exit(84)
    if i == 0:
      training = numpy.array(data.iloc[id, :], dtype = float)
    else:
      training = numpy.append(training, numpy.array(data.iloc[id, :], dtype = float))
    i += 1
training = numpy.reshape(training, (number_of_data, -1))

model = Sequential()
model.add(Dense(units = 4000, activation = 'sigmoid', input_dim = 4231))
model.add(BatchNormalization())
model.add(Dense(units = 5000, activation = 'sigmoid'))
model.add(Dense(units = 3000, kernel_initializer = 'he_normal', activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.08))
model.add(Dense(units = 3500, kernel_initializer = 'he_normal', activation = 'relu'))
model.add(Dense(units = 2000, kernel_initializer = 'he_normal', activation = 'relu'))
model.add(BatchNormalization())
model.add(Dense(units = 400, kernel_initializer = 'he_normal', activation = 'relu'))
model.add(Dense(units = 100, kernel_initializer = 'he_normal', activation = 'relu'))
model.add(BatchNormalization())
model.add(Dense(units = 40, kernel_initializer = 'he_normal', activation = 'relu'))
model.add(Dropout(0.04))
model.add(Dense(units = 10, kernel_initializer = 'he_normal', activation = 'relu'))
model.add(BatchNormalization())
model.add(Dense(units = 6, kernel_initializer = 'he_normal', activation = 'relu'))
model.add(Dense(units = 4, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = SGD(lr = 0.01, decay = 1e-6,
              momentum = 0.9, nesterov = True), metrics = ['accuracy'])
print(model.summary)
model.fit(training, teacher, epochs = number_of_epochs, validation_split = 0.2,
              batch_size = number_of_batch_size)

predicted = model.predict(testing)

with open(output_results, 'a') as output_result:
  print(exclude, end = "\t", file = output_result)
  if answer[0, 0] == 1:
    print('YRI', end = "\t", file = output_result)
  elif answer[0, 1] == 1:
    print('CEU', end = "\t", file = output_result)
  elif answer[0, 2] == 1:
    print('CHB', end = "\t", file = output_result)
  elif answer[0, 3] == 1:
    print('JPT', end = "\t", file = output_result)
  else:
    sys.stderr.write('Error: no answer ' + str(answer) + "\n")
    sys.exit(86)
  for i in range(populations):
    print('{:.5f}'.format(predicted[0][i]), end = "\t", file = output_result)
  if answer[0, numpy.argmax(predicted[0])] == 1:
    print('correct', file = output_result)
  else:
    print('wrong', file = output_result)
