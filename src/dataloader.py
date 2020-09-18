import pandas as pd
import sys
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from os.path import join
from PIL import Image
from transformations import Transformations

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
		self.main_path = datapath
		self.RGB = rgb

		# Enable data augmentation only if training. The user can decide to
		# remove it though
		if self.state == 'train':
			self.data_aug = data_aug
		else:
			self.data_aug = False

		## PATHS TO IMAGES
		self.img_fold = join(self.main_path, 'img_align_celeba')
		self.eval_file = join(self.main_path, 'list_eval_partition.csv')

		## READ THE CSV
		df_eval = pd.read_csv(self.eval_file)

		## FILTERS
		filt_train = (df_eval['partition'] == 0)
		filt_test = (df_eval['partition'] == 2)

		## DATASETS FOR TRAINING AND TESTING
		df_train = df_eval.loc[filt_train]
		df_test = df_eval.loc[filt_test]

		## FILL LIST ACCORDING TO TRAIN/TEST PHASE
		if self.state == 'train':
			# define image list
			self.img_list = df_train['image_id'].to_list()

		if state == 'test':
			# define image list
			self.img_list = df_test['image_id'].to_list()

		## DEFINE PRE-PROC ACCORDING TO INPUT
		self.transf = Transformations(self.data_aug, self.RGB)

		# debug
		del filt_train, filt_test, df_eval, df_train, df_test


	def __getitem__(self, index):
		"""built-in

		returns either the test or train images and labels
		"""

		# strings
		img_path = join(self.img_fold, self.img_list[index])

		# open image and make it tensor
		target = self.transf.perform(img_path)

		return target

	def __len__(self):
		return len(self.img_list)