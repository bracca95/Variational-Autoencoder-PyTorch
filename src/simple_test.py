import os
import cv2
import math
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms 
from torchvision import utils
from PIL import Image
from tqdm import tqdm

class TestSimple:
	def __init__(self, device):
		""" init

		if isplain_test == True the program performs plain testing,
		otherwise it performs latent space transition
		"""

		self.device = device
		self.full_name_list = None			# type list from df
		self.new_path = '../data/test'


	def read(self):
		""" read

		the function reads the test images (partition 2) and apply the VAE on 
		those images.
		"""

		folder = '../dataset/img_align_celeba'
		folder = os.path.abspath(os.path.expanduser(folder))

		csv_partition = 'list_eval_partition.csv'
		csv_path = os.path.join(os.path.dirname(folder), csv_partition)

		df = pd.read_csv(csv_path)

		filt = (df['partition'] == 2)
		df_test = df.loc[filt]

		name_list = df_test['image_id'].to_list()
		full_name_list = [os.path.join(folder, name) for name in name_list]

		# keep during debugging to compare memory usage
		del df, df_test
		self.full_name_list = full_name_list

	
	def write(self):

		""" write

		write the desired images in the ../data/test folder
		"""

		if self.full_name_list is None:
			self.read()

		totensor = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
			])

		preprocess = transforms.Compose([
				transforms.CenterCrop((178, 178)),
				transforms.CenterCrop((128, 128))
			])

		for img_name in self.full_name_list:
			name = os.path.basename(img_name)
			img = Image.open(img_name).convert('RGB')
			img = preprocess(img)
			img = totensor(img)
			input_image = img.unsqueeze(0).to(self.device)
			utils.save_image(0.5*input_image+0.5, os.path.join(self.new_path, name))

		# keep during debugging to compare memory usage
		del self.full_name_list, img, input_image