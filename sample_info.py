#!/usr/bin/env python

# sample_info.py

import pandas

excel_file = pandas.ExcelFile('20130606_sample_info.xlsx')
samples = excel_file.parse(excel_file.sheet_names[0])
for i in range(len(samples)):
  if samples.iat[i, 2] == 'YRI' or \
     samples.iat[i, 2] == 'CEU' or \
     samples.iat[i, 2] == 'CHB' or \
     samples.iat[i, 2] == 'JPT':
    print(samples.iat[i, 0], samples.iat[i, 2], sep = "\t")
