#!/usr/bin/env python3

# crop_faces.py - crop faces using position data
# coded   by Kohji on 2019-12-15
# updated by Kohji on 2019-12-19

import os
import sys
import cv2

if len(sys.argv) < 2:
  sys.stderr.write('Error 12: An input text file is requied' +
                   ' as the first command argument.' + "\n")
  sys.exit(12)

fixed_width = 90

counter = 0
with open(sys.argv[1]) as tsv:
  for line in tsv:
    fields = line[:-1].split("\t")
    file_jpeg = '../renamed/' + fields[0]
    if not os.path.isfile(file_jpeg):
      sys.stderr.write('Error 14: File not found ' + file_jpeg + "\n")
      sys.exit(12)
    photo = cv2.imread(file_jpeg)
    x = int(fields[1])
    y = int(fields[2])
    w = int(fields[3])
    h = int(fields[4])
    crop = photo[y:(y + h), x:(x + w)]
    resized = cv2.resize(crop, (fixed_width, int(h * fixed_width / w)))
    counter += 1
    cv2.imwrite('crop_' + fields[0], resized)
