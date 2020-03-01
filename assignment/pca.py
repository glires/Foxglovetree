import pandas
import matplotlib.pyplot
import sklearn.decomposition

data_file_name = 'data.tsv'
dim = 8

data_frame = pandas.read_csv(data_file_name, sep = "\t", index_col = 0)
individuals = []
for i in range(len(data_frame)):
  if (data_frame.index[i])[8:] == 'CEU':
    population = 3
  elif (data_frame.index[i])[8:] == 'CHB':
    population = 0
  elif (data_frame.index[i])[8:] == 'JPT':
    population = 2
  elif (data_frame.index[i])[8:] == 'YRI':
    population = 1
  else:
    population = 9
  individuals.append(population)

pca = sklearn.decomposition.PCA(n_components = dim)
pca.fit(data_frame)
feature = pca.transform(data_frame)
print(pandas.DataFrame(feature, columns = ["PC{}".format(x + 1) for x in range(dim)]).head())

matplotlib.pyplot.scatter(feature[:, 0], feature[:, 1], s = 2, alpha = 1.0, c = individuals)
matplotlib.pyplot.xlabel('PC1')
matplotlib.pyplot.ylabel('PC2')
matplotlib.pyplot.savefig('pca.pdf', bbox_inches = 'tight')
