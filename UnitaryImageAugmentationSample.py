# This script use Keras to create multiple images from a single one
# you can tweak ImageDatagenerator in order to have different results
import os

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = load_img('data/train/refImage.png')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `generated/` directory
# please change image_class_prefix to a label corresponding to your class
# ensure directory exists
i = 0
image_class_prefix = 'MY_CLASS'
dest_dir = 'data/generated'

if not os.path.exists(dest_dir) :
        os.makedirs(dest_dir)

for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=dest_dir, save_prefix=image_class_prefix, save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely