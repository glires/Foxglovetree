#!/usr/bin/env python
# clustering.py -- assignment Oct 2019
# coded by Kohji

import pandas
import matplotlib.pyplot
from scipy.cluster.hierarchy import dendrogram, linkage
%matplotlib inline

data_frame = pandas.read_csv('data.tsv', sep = "\t", index_col = 0)
samples = data_frame.index
result = linkage(data_frame, method = 'ward', metric = 'euclidean')
matplotlib.rcParams['lines.linewidth'] = 0.2
dendrogram(result, labels = samples)
matplotlib.pyplot.xticks(rotation = 90)
matplotlib.pyplot.tick_params(axis = 'x', which = 'major', labelsize = 1)
matplotlib.pyplot.tick_params(axis = 'y', which = 'major', labelsize = 8)
matplotlib.pyplot.title('Dendrogram of 414 individuals', fontsize = 8)
matplotlib.pyplot.xlabel('Individual', fontsize = 8)
matplotlib.pyplot.savefig('assign_10_dendrogram.pdf', bbox_inches = 'tight')
matplotlib.pyplot.show()
