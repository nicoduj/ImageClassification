# Image Classification tests scripts with Keras and tensorflow

This work was based on the following Keras blog Post : 
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html


The purpose was to evaluate image Classification with a small data set, and compare it with an SVM approach.
However, and unlike the blog post, I wanted to evaluate this approcah on a multi class problem.

I won't share the business case of this work, only the models that were used, and some scripts made to evaluate image Augmentation impact on those.

Feel free to take those as a first step, and improove them by trying and error. I am no data scientist at all, the purpose there was just to experiment and see the tools in action :see_no_evil: :hear_no_evil: :speak_no_evil: 


## TensorFlow Environment installation

You will find complete installation instructions there : 
https://www.tensorflow.org/install/install_mac

As advised, I used the following steps to build my virtualEnv installation on a mac : 

1. Start a terminal (a shell). You'll perform all subsequent steps in this shell.

2. Install pip and Virtualenv :

```bash
 $ sudo easy_install pip
 $ pip install --upgrade virtualenv 
 ```

3. Create a Virtualenv environment :

```bash
 $ virtualenv --system-site-packages -p python3 targetDirectory 
```

where targetDirectory identifies the top of the Virtualenv tree.

4. Activate the Virtualenv environment :

```bash
$ cd targetDirectory
$ source ./bin/activate      # If using bash, sh, ksh, or zsh
```

The preceding source command should change your prompt to the following:

```bash
 (targetDirectory)$ 
```

5. Ensure pip ≥8.1 is installed:
```bash
 (targetDirectory)$ easy_install -U pip
 ```

6. Issue the following command to install TensorFlow and all the packages that TensorFlow requires into the active Virtualenv environment:

```bash
 (targetDirectory)$ pip3 install --upgrade tensorflow 
 ```

## Keras installation

Inside your virtualenv :

```bash
 (targetDirectory)$ pip3 install keras 
```

You will also need pillow for image modification and h5py for saving models for thoses scripts to run :

```bash
 (targetDirectory)$ pip3 install Pillow
 (targetDirectory)$ pip3 install h5py
```

## Tensorboard installation

Using Tensorboard was a good way to have a visual comparison between iterations. 

Inside your virtualenv :

```bash
 (targetDirectory)$ pip3 install tensorboard 
 ```

Be aware that in order to launch Tensorboard, you to do it from tensoarboard directory directly like this :

```bash
$ cd targetDirectory/lib/python3.6/site-packages/tensorboard/
$ tensorboard --logdir ~/[YOUR_LOG_DIR]
 ```

Then go to tensorbaord URL with your favorite browser :  http://localhost:6006 


## Scripts description and usage

For all those samples, please ensure that directories exist before launching scripts.

### Image augmentation

1. [UnitaryImageAugmentationSample](UnitaryImageAugmentationSample.py)
This sript demonstrate a simple image augmentation method from one image

2. [ClassImageAugmentationSample](ClassImageAugmentationSample.py)
This script use the same approcah, but directory based, so you can make equivalent classes in temr of numbers of sample .


### Models

1. [FirstModel](FirstModel.py)
A model that use dropout only on the activation layer, and image augmentation technique

<p align="center">
<img src="https://user-images.githubusercontent.com/19813688/36601211-e2e09176-18b4-11e8-80cf-6340e1561482.png" 
width="300" alt="run 1519394356">
</p>

2. [SecondModel](SecondModel.py)
A model that use dropout on each layer, and class weights representation. No augmentation on this test.

<p align="center">
<img src="https://user-images.githubusercontent.com/19813688/36601216-e5c048f0-18b4-11e8-9187-6b2cec2402a9.png" 
width="300" alt="run 1519397495">
</p>

## Results

The results on my test data, composed of 341 images belonging to 4 classes for training and 87 images belonging to 4 classes for validation, are the following after 50 epochs :
- FirstModel (in <span style="color:rgb(0, 119, 187)">Blue</span>) : loss: 0.7886 - acc: 0.7267 - val_loss: 0.4058 - val_acc: 0.8800 
- SecondModel (in <span style="color:rgb(204, 51, 17)">Red</span>) : loss: 0.0470 - acc: 0.9867 - val_loss: 1.8659 - val_acc: 0.8200 

(Runs took 46mn / 48 mn on a macbook air 1,3 GHz Intel Core i5)

<p align="center">
<img src="https://user-images.githubusercontent.com/19813688/36602723-0b1e30f4-18b9-11e8-8e6e-7e86f79594d4.png" 
width="300" alt="Acc">
<img src="https://user-images.githubusercontent.com/19813688/36602722-0b05b862-18b9-11e8-8367-05269b30f6e2.png" 
width="300" alt="Loss">
</p>
<p align="center">
<img src="https://user-images.githubusercontent.com/19813688/36602720-0acfbbae-18b9-11e8-9251-3f2078283f91.png" 
width="300" alt="Val_acc">
<img src="https://user-images.githubusercontent.com/19813688/36602721-0ae97486-18b9-11e8-8895-2b523c3077d8.png" 
width="300" alt="Val_loss">
</p>

As you can see, the second approach gives better theorical results, but loss is very fuzzy on validation set, meaning that the model is not stable at all from my point of vue.

## Reference

As mentionned, this work is mainly based on the following blog post : 
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html


## Author

Nicolas Dujardin, nicolas.dujardin@gmail.com

## License

Those scripts are  available under the MIT license. See the LICENSE file for more info.

