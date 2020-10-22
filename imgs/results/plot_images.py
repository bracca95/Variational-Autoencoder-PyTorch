import os
import cv2
import matplotlib.pyplot as plt

acc_ext = ('.jpg', '.jpeg', '.png')

folder = '.'
full_fol = os.path.abspath(os.path.expanduser(folder))

lr_images = []
sr_images = []
hr_images = []

for img_name in sorted(os.listdir(full_fol)):
	if img_name.endswith(acc_ext):
		
		if 'blur' in img_name:
			img = cv2.imread(os.path.join(full_fol, img_name))
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			lr_images.append(img)

		if 'recon' in img_name:
			img = cv2.imread(os.path.join(full_fol, img_name))
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			sr_images.append(img)

		if 'orig' in img_name:
			img = cv2.imread(os.path.join(full_fol, img_name))
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			hr_images.append(img)

line1 = [lr_images[0], sr_images[0], hr_images[0]]
line2 = [lr_images[1], sr_images[1], hr_images[1]]
line3 = [lr_images[2], sr_images[2], hr_images[2]]
line4 = [lr_images[3], sr_images[3], hr_images[3]]

# plot
Nr = 4
Nc = 3

fig, axs = plt.subplots(Nr, Nc, figsize=(10, 10))
#fig.suptitle('neymar (w/ training on him)')

images = []
for i in range(Nr):
	for j in range(Nc):
		if i == 0:
			images.append(axs[i, j].imshow(line1[j]))
		if i == 1:
			images.append(axs[i, j].imshow(line2[j]))
		if i == 2:
			images.append(axs[i, j].imshow(line3[j]))
		if i == 3:
			images.append(axs[i, j].imshow(line4[j]))	


plt.savefig('train_on_neymar.png')
plt.show()

## debug
# cv2.imshow('window', hr_images[0])
# cv2.waitKey(0)
