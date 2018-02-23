################################
#in order to launch tensoarboard
################################
#cd ~/tensorflow/lib/python3.6/site-packages/tensorboard/
#python tensorboard --logdir ~/Documents/DEV/TEST_DL_CLASSIF/keras/logs/
################################


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import TensorBoard
from time import time
from PIL import Image
from collections import Counter
from keras.optimizers import RMSprop
import numpy as np



# setup parameters
Image.MAX_IMAGE_PIXELS = 1000000000 
img_width, img_height = 150, 150
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 250
nb_validation_samples = 50
epochs = 25
batch_size = 10
#use different number if you want to restart from a specific epochs
start_epoch = 0 


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


#no augmentation there, we assume the data was augmented before through the scripts provided
train_datagen = ImageDataGenerator(rescale=1./255)


dimData = np.prod(input_shape[1:])

# Our model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(Dropout(0.10))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.10))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.10))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.25))
# PLEASE MODIFY DEPENDING OF THE NUMBER OF CLASSES YOU HAVE
model.add(Dense(4))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


#printing model summary            
model.summary()

# our train generator
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


counter = Counter(train_generator.classes)                          
max_val = float(max(counter.values()))       
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}                     

# printing class weights
print(class_weights)

# tensoarboard output
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

# fitting the model
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples/batch_size,
    epochs=epochs,
    class_weight=class_weights,
    verbose=1,
    callbacks=[tensorboard],
    initial_epoch=start_epoch
    )

model.save('SecondModel.h5')
model.save_weights('SecondModelWeights.h5')
