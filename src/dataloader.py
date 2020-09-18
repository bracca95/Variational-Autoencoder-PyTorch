import os
import pandas as pd
import sys
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from PIL import Image
from transformations import Transformations

accept_ext = ('.jpg', '.jpeg', '.png')

class CelebDataSet(Dataset):
	"""CelebA Dataset"""

	def __init__(self, state, data_aug, rgb, datapath='../dataset'):
		"""init parameters
		
		init parameters

		state must be specified either as 'train' or 'test'.
		Data augmentation is suggested during training (active by default)
		rgb = 0 -> B/N
		rgb = 1 -> rgb
		"""
		super(CelebDataSet, self).__init__()

		if state == None:
			sys.exit('specify a state, either \'train\' or \'test\'')

		## INIT
		self.state = state
		self.main_path = os.path.abspath(os.path.expanduser(datapath))
		self.RGB = rgb

		# Enable data augmentation only if training. The user can decide to
		# remove it though
		if self.state == 'train':
			self.data_aug = data_aug
		else:
			self.data_aug = False

		## PATHS TO IMAGES
		file_list = os.listdir(self.main_path)
		self.img_list = []

		for file in file_list:
			if file.endswith(accept_ext):
				self.img_list.append(os.path.join(self.main_path, file))


		## DEFINE PRE-PROC ACCORDING TO INPUT
		self.transf = Transformations(self.data_aug, self.RGB)


	def __getitem__(self, index):
		"""built-in

		returns either the test or train images and labels
		"""

		# strings
		img_path = self.img_list[index]

		# open image and make it tensor
		target = self.transf.perform(img_path)

		return target

	def __len__(self):
		return len(self.img_list)