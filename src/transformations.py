import torchvision.transforms as transforms
import pandas as pd
from os.path import join
from PIL import Image       # PIL is currently being developed as Pillow

class Transformations:
	def __init__(self, data_aug, rgb):

		self.data_aug = data_aug
		self.RGB = rgb

		## DEFINE PRE-PROCESSING STRATEGY
		if self.data_aug == False:
			self.pre_process = transforms.Compose([
								transforms.CenterCrop((178, 178)),
								transforms.CenterCrop((128,128)),
								])
		else:
			self.pre_process = transforms.Compose([
					transforms.RandomHorizontalFlip(),
					transforms.CenterCrop((178, 178)),
					transforms.CenterCrop((128, 128)),
					transforms.RandomRotation(20, resample=Image.BILINEAR),
					transforms.ColorJitter(brightness=0.4, 
											contrast=0.4, 
											saturation=0.4, 
											hue=0.1)
					])

		if self.RGB == 0:
			self.totensor = transforms.Compose([
							transforms.ToTensor(),
							transforms.Normalize([0.5], [0.5])
							])
		else:
			self.totensor = transforms.Compose([
							transforms.ToTensor(),
							transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
							])



	def perform(self, image_path):
		
		# this shall be your benchmark
		if self.RGB == 0: target_image = Image.open(image_path).convert('L')
		else: target_image = Image.open(image_path).convert('RGB')

		# normalize all images
		target_image = self.totensor(target_image)

		return target_image