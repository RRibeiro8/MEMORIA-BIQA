import os
import pandas as pd

import torch
from torch.utils.data import Dataset

import torchvision
from torchvision import datasets, models, transforms

from PIL import Image

class Koniq_10k_Dataset(Dataset):

	def __init__(self, transform=None, method='train'):
		
		ids = pd.read_csv('datasets/koniq-10k/koniq10k_distributions_sets.csv')
		if method == 'train':
			self.ids = ids[ids.set=='training']
		elif method == 'val':
			self.ids = ids[ids.set=='validation']
		else:
			self.ids = ids

		self.img_dir='datasets/koniq-10k/512x384/'
		self.transform = transform

	def __len__(self):
		return len(self.ids)

	def __getitem__(self, idx):
		img_path = os.path.join(self.img_dir, self.ids.iloc[idx, 0])
		image = Image.open(img_path)
		label = self.ids.iloc[idx, 1]
		if self.transform:
			image = self.transform(image)

		return image, label


class DataLoader(object):

	def __init__(self, batch_size=1):

		self.batch_size = batch_size

		self.data_transforms = {
			'train': transforms.Compose([
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
			]),
			'val': transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
			]),
		}

	def get_loader(self):

		dataloaders = {x: torch.utils.data.DataLoader(Koniq_10k_Dataset(transform=self.data_transforms[x], method=x), batch_size=8,
											 shuffle=True, num_workers=4)
					for x in ['train', 'val']}

		dataset_sizes = {x: len(Koniq_10k_Dataset(transform=self.data_transforms[x], method=x)) for x in ['train', 'val']}

		return dataloaders, dataset_sizes

