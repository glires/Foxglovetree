#!/usr/bin/env python3

'''
cross_validation.py - perform classification of the Foxglovetree data
'''

__author__  = 'Kohji OKAMURA, Ph.D.'
__date__    = '2025-10-24'
__version__ = 0.5

import sys
import numpy
import torch
import datetime
import torchvision

npy_images = './data/headshot_data.npy'
npy_labels = './data/headshot_labels.npy'

n_epochs = 50	# can be changed using sys.argv[1]
learning_rate = 0.03
momentum = 0.9
weight_decay = 1e-4
size_npy = 359	# Foxglovetree size
size_iv3 = 299	# Inception V3
size_rgb = 3	# number of channels
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)
gender = ('m', 'f')
val_split = 0.2
batch_t = 64
batch_v = 32
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class ImageTransform():
  def __init__(self, resize, mean, std):
    self.data_transform = {
        'train': torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(p = 0.4),
            torchvision.transforms.RandomAffine(degrees = (-9, 9),
                scale = (0.9, 1.1)),
            torchvision.transforms.RandomErasing(p = 0.2,
                scale = (0.01, 0.05), ratio=(0.8, 1.2)),
            torchvision.transforms.RandomPerspective(p = 0.2),
            torchvision.transforms.ColorJitter(brightness = (0.2, 1.7),
                contrast = 0.2, saturation = 0.1, hue = 0.1),
            torchvision.transforms.RandomAutocontrast(p = 0.2),
            torchvision.transforms.RandomRotation(degrees = 4),
            torchvision.transforms.Resize((resize, resize)),
            torchvision.transforms.Normalize(mean, std)]),
        'valid': torchvision.transforms.Compose([
            torchvision.transforms.Resize((resize, resize)),
            torchvision.transforms.Normalize(mean, std)]) }

  def __call__(self, img, phase = 'train'):
    return self.data_transform[phase](img)

class FoxglovetreeDataset(torch.utils.data.Dataset):
  def __init__(self, list_images, classes, transform = None, phase = 'train'):
    self.list_images = list_images
    self.transform = transform
    self.classes = classes
    self.phase = phase

  def __len__(self):
    return len(self.list_images)

  def __getitem__(self, index):
    image_npy = numpy.empty((size_rgb, size_iv3, size_iv3))
    margin = (size_npy - size_iv3) // 2
    shot = self.list_images[index]
    image_iv3 = images[shot].reshape(size_npy, -1)[margin:(margin + size_iv3),
                                                   margin:(margin + size_iv3)]
    for c in range(size_rgb): image_npy[c, ...] = image_iv3
    img = torch.from_numpy(image_npy / 255.0).float().to(device)
    img_transformed = self.transform(img, self.phase)
    label = labels[shot][2]
    return img_transformed, int(label)

if __name__ == '__main__':
  current = datetime.datetime.now()

  try:    epochs = int(sys.argv[1])
  except: epochs = n_epochs

  images = numpy.load(npy_images)
  labels = numpy.load(npy_labels)
  n = len(labels)	# must be the number of 277-volunteer headshot photos, 2429
  fs = set()
  ms = set()
  for shot in range(n):
    if labels[shot][2] == 1: fs.add(labels[shot][0])
    else:                    ms.add(labels[shot][0])
  f_val = numpy.random.choice(numpy.array(list(fs)), int(len(fs) * val_split), replace = False)
  m_val = numpy.random.choice(numpy.array(list(ms)), int(len(ms) * val_split), replace = False)
  vals = numpy.append(f_val, m_val)
  del fs, ms, f_val, m_val

  images_train = []
  images_valid = []
  for shot in range(n):
    if labels[shot][0] in vals: images_valid.append(shot)
    else:                       images_train.append(shot)

  train_dataset = FoxglovetreeDataset(list_images = images_train, classes = gender,
                      transform = ImageTransform(size_iv3, mean, std), phase = 'train')
  valid_dataset = FoxglovetreeDataset(list_images = images_valid, classes = gender,
                      transform = ImageTransform(size_iv3, mean, std), phase = 'valid')
  train_dataloader = torch.utils.data.DataLoader(train_dataset,
                         batch_size = batch_t, shuffle = True)
  valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
                         batch_size = batch_v, shuffle = False)
  dataloaders_dict = { 'train': train_dataloader, 'valid': valid_dataloader }

  cnn = torchvision.models.inception_v3(weights = 'Inception_V3_Weights.IMAGENET1K_V1').to(device)
  cnn.fc = torch.nn.Linear(cnn.fc.in_features, len(gender)).to(device)
  cnn.aux_logits, cnn.AuxLogits = False, None
        # cross_entropy_loss(): argument 'input' (position 1) must be Tensor, not InceptionOutputs
  criterion = torch.nn.CrossEntropyLoss()

  optimizer = torch.optim.SGD(cnn.parameters(),
                  lr = learning_rate, momentum = momentum, weight_decay = weight_decay)

  for epoch in range(epochs):
    print(f'Epoch {epoch + 1:02d}/{epochs:02d}')

    for phase in ('train', 'valid'):
      if phase == 'train':
        cnn.train()
      else:
        cnn.eval()

      epoch_loss = 0.0
      epoch_corrects = 0
      for inputs, labl in dataloaders_dict[phase]:
        labl = labl.to(torch.uint8).to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(phase == 'train'):
          outputs = cnn(inputs)
          loss = criterion(outputs, labl)
          preds = torch.max(outputs, 1)[1]
          if phase == 'train':
            loss.backward(retain_graph = True)
            optimizer.step()
          epoch_loss += loss.item() * inputs.size(0)
          epoch_corrects += torch.sum(preds == labl.data)

      epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
      epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)
      print(f'  {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

  # torch.save(cnn, model)	# model = './data/foxglovetree.pt'

  minutes, seconds = divmod(int((datetime.datetime.now() - current).total_seconds()), 60)
  print(f'Time elapsed: {minutes:02d}:{seconds:02d}')
