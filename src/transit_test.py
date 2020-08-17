import os
import cv2
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from tqdm import tqdm
from PIL import Image

class Transition:
	def __init__(self, device):
		self.device = device
		self.df_compare = None
		self.output_dir = '../data/test'
		self.csv_path = '../last_csv.csv'

	
	def return_pair_list(self):
		
		# check if there images are already present
		# https://stackoverflow.com/a/33400758
		if not any(fname.endswith('.jpg') for fname in os.listdir(self.output_dir)):
			last_csv = self.processing(self.output_dir, self.csv_path)
		else:
			last_csv = pd.read_csv(self.csv_path)
		
		images_list = []
		df_len = math.floor(len(last_csv)/2)

		for i in range(df_len):
			try:
				img1_path = last_csv.iloc[2*i]['abspath']
				img2_path = last_csv.iloc[2*i + 1]['abspath']
				img1 = Image.open(img1_path).convert('RGB')
				img2 = Image.open(img2_path).convert('RGB')
				images_list.append((img1, img2))
			except IndexError:
				# debug
				print(i)

		return images_list


	def processing(self, output_dir, csv_path):

		if self.df_compare == None:
			self.get_df_compare()

		csv_path = os.path.abspath(os.path.expanduser(csv_path))

		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		df_label_list = list(self.df_compare['label'].unique())
		df_new = pd.DataFrame([], columns=['path', 'abspath', 'label', 'name'])
		
		for i, val in enumerate(tqdm(df_label_list)):
			filt = (self.df_compare['label'] == val)
			df_pair = self.df_compare.loc[filt]

			# full path
			img_bench_path = df_pair.iloc[0]['abspath']
			img_modif_path = df_pair.iloc[1]['abspath']

			# name
			img_bench_name = os.path.basename(img_bench_path)
			img_modif_name = os.path.basename(img_modif_path)

			# output path
			img_bench_out_path = join(output_dir, img_bench_name)
			img_modif_out_path = join(output_dir, img_modif_name)

			# new df
			static_img1 = df_pair.iloc[0][['path', 'label', 'name']]
			static_img1['abspath'] = img_bench_out_path
			static_img1 = static_img1[['path', 'abspath', 'label', 'name']]

			static_img2 = df_pair.iloc[1][['path', 'label', 'name']]
			static_img2['abspath'] = img_modif_out_path
			static_img2 = static_img2[['path', 'abspath', 'label', 'name']]

			df_new = df_new.append(static_img1, ignore_index=True)
			df_new = df_new.append(static_img2, ignore_index=True)

			self.write_img(img_bench_path, img_bench_out_path, upscale=False)
			self.write_img(img_modif_path, img_modif_out_path, upscale=True)

		df_new.to_csv(csv_path)
		return df_new


	def write_img(self, img_path, output_path, upscale):

		img_colo = cv2.imread(img_path, cv2.IMREAD_COLOR)
		img_gray = cv2.cvtColor(img_colo, cv2.COLOR_BGR2GRAY)

		if upscale == False:

			rows, cols, _ = map(int, img_colo.shape)

			old_shape = (rows, cols)	# h (row), w (col)
			new_shape = (128, 128)	   # h (row), w (col)
			img = self.center_crop(img_colo, old_shape, new_shape)

			# normalization
			img_template = np.zeros(new_shape)
			img_norm = cv2.normalize(img, img_template, 0, 255, cv2.NORM_MINMAX)

			## OUTPUT
			cv2.imwrite(output_path, img_norm)
		
		else:
			rows, cols, _ = map(int, img_colo.shape)

			old_shape = (rows, cols)
			new_shape = (112, 96)
			hr_shape = (128, 128)
			lr_shape = (16, 16)
			nTimes = int(math.log2(hr_shape[0]/lr_shape[0]))

			img = self.center_crop(img_colo, old_shape, hr_shape)

			# normalization
			img_template = np.zeros(hr_shape)
			img_norm = cv2.normalize(img, img_template, 0, 255, cv2.NORM_MINMAX)

			# resize to (16, 16)
			img_resz_lr = cv2.resize(img_norm, lr_shape)
			row, col, _ = map(int, img_resz_lr.shape)

			# upscale to (128, 128)
			for i in range(nTimes):
				img_resz_hr = cv2.pyrUp(img_resz_lr, dstsize=(2*row, 2*col))

				# update values
				img_resz_lr = img_resz_hr
				row, col, _ = map(int, img_resz_lr.shape)

			cv2.imwrite(output_path, img_resz_lr)


	# https://progr.interplanety.org/en/python-how-to-find-the-polygon-center-coordinates/
	def center_crop(self, img, old_shape, new_shape):
		assert type(old_shape)==tuple, 'shape must be of type tuple'
		assert type(new_shape)==tuple, 'shape must be of type tuple'

		p1 = (0, 0)
		p2 = (0, old_shape[1])
		p3 = (old_shape[0], old_shape[1])
		p4 = (old_shape[0], 0)

		vertexes = (p1, p2, p3, p4)

		x_list = [vertex [0] for vertex in vertexes]
		y_list = [vertex [1] for vertex in vertexes]
		n_vert = len(vertexes)
		
		x = sum(x_list) / n_vert
		y = sum(y_list) / n_vert

		cent = (x, y)

		# upper left point cx = w + dw/2, cy = h + dh/2
		ulp_h = math.floor(cent[0] - (new_shape[0]/2))
		ulp_w = math.floor(cent[1] - (new_shape[1]/2))

		img = img[ulp_h:ulp_h+new_shape[0], ulp_w:ulp_w+new_shape[1]]
		
		return img


	def get_df_compare(self):
		
		folder = '/content/Variational-Autoencoder-PyTorch/dataset/img_align_celeba'

		folder = os.path.abspath(os.path.expanduser(folder))
		csv_partition = 'list_eval_partition.csv'
		csv_identity = 'identity_CelebA.csv'

		csv_partition_path = os.path.join(os.path.dirname(folder), csv_partition)
		csv_identity_path = os.path.join(os.path.dirname(folder), csv_identity)

		df_partition = pd.read_csv(csv_partition_path)
		df_identity = pd.read_csv(csv_identity_path)

		df = df_partition.copy()

		# this can be done because indexing is the same 
		# https://stackoverflow.com/a/45747631
		# now we have a dataset with ['image_id']['partition']['identity']
		df['identity'] = df_identity['identity']

		# format dataframe as the program requires
		# https://stackoverflow.com/a/20027386
		# https://stackoverflow.com/a/53816799
		df_full = pd.DataFrame({
			'path': df['image_id'],
			'abspath': folder + '/' + df['image_id'].astype(str),
			'label': df['identity'],
			'name': df['identity']
			})

		## REMOVE IDENTITIES THAT ONLY OCCUR ONCE
		df_group = df_full.groupby(['label']).count()

		filt1 = (df_group['path']==1)
		df_excptn = df_group.loc[filt1]
		df_excptn.reset_index(inplace=True)
		df_excptn = df_excptn[['path', 'abspath', 'label', 'name']]

		filt2 = (df_full['label'].isin(df_excptn['label']) == False)
		df_full = df_full.loc[filt2]

		## GET ONLY TWO IMAGES FOR IDENTITY
		df_min = df_full.groupby(['label']).min()
		df_min.reset_index(inplace=True)
		df_min = df_min[['path', 'abspath', 'label', 'name']]

		df_max = df_full.groupby(['label']).max()
		df_max.reset_index(inplace=True)
		df_max = df_max[['path', 'abspath', 'label', 'name']]

		filt = (df_full['path'].isin(df_min['path']) == True) | \
				(df_full['path'].isin(df_max['path']) == True)
		df_compare = df_full.loc[filt]

		del df_partition, df_identity, df, df_full, df_group, df_excptn, \
			df_min, df_max

		self.df_compare = df_compare