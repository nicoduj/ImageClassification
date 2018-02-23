# This script use Keras to create multiple images from a batch of image per class
# its purpose is to generate completion on classes in order them to have the same number of samples.
# you can tweak ImageDatagenerator in order to have different results


from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import random
import os



#the function for class augmentation
def generate(input_path = '',output_path = '', target_nb = 1000):
	count = 10
	fileOutput = '{}/Random_{}.jpg'



	for class_name in os.listdir(input_path.format('')) :

		if not class_name.startswith('.'):
		
			print ('Generating Training data for ', class_name)
	
			if not os.path.exists(output_path.format(class_name)) :
				os.makedirs(output_path.format(class_name))


			nbfiles = len([name for name in os.listdir(input_path.format(class_name)) if os.path.isfile(os.path.join(input_path.format(class_name), name)) and not name.startswith('.')])
			print ('NB existing images :', nbfiles)
			nbfilesRandom = len([name for name in os.listdir(output_path.format(class_name)) if os.path.isfile(os.path.join(output_path.format(class_name), name)) and not name.startswith('.')])
			total =  target_nb - (nbfiles+nbfilesRandom)
			print ('NB image to generate :', total)

			 
			gen = ImageDataGenerator(
				rotation_range=10,
				width_shift_range=0.1,
				height_shift_range=0.1,
				shear_range=0.1,
				zoom_range=0.1,
				horizontal_flip=False,
				fill_mode='nearest')

			j=nbfilesRandom
	
			while (j-nbfilesRandom)<total:
				# load image to array
				input_file = os.path.join(input_path.format(class_name),random.choice( [name for name in os.listdir(input_path.format(class_name)) if not name.startswith('.') ]) )
				print(input_file)
				image = img_to_array(load_img(input_file))

				# reshape to array rank 4
				image = image.reshape((1,) + image.shape)

				# let's create infinite flow of images
				images_flow = gen.flow(image, batch_size=1)
	
				for i, new_images in enumerate(images_flow):
					# we access only first image because of batch_size=1
					new_image = array_to_img(new_images[0], scale=True)
					new_image.save(output_path.format(fileOutput.format(class_name,j*1000 + i + 1)))
					print ('step:',i,j)
					j = j+1
					if i >= count:
						break
					if j >= total:
						break



#train directory to augment 
input_path = 'data/train/{}'
output_path = 'data/generated/train/{}'
generate(input_path,output_path,250)

#validation directory to augment
input_path = 'data/validation/{}'
output_path = 'data/generated/validation/{}'
generate(input_path,output_path,50)
        
