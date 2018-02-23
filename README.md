# Image Classification tests scripts with Keras and tensorflow

This work was based on the following Keras blog Post : 
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html


The purpose was to evaluate image Classification with a small data set, and compare it with an SVM approach.
However, and unlike the blog post, I wanted to evaluate this approcah on a multi class problem.

I won't share the business case of this work, only the models that were used, and some scripts made to avaluate image Augmentation impact on those.

Feel free to take those as a first steps, and improove them by trying and error :see_no_evil: :hear_no_evil: :speak_no_evil: 


## TensorFlow Environment installation

You will find all installation option there : 
https://www.tensorflow.org/install/install_mac

As advised, I used the following steps to build my virtualEnv installation on a mac : 

1. Start a terminal (a shell). You'll perform all subsequent steps in this shell.

2. Install pip and Virtualenv by issuing the following commands:

```bash
 $ sudo easy_install pip
 $ pip install --upgrade virtualenv 
 ```

3. Create a Virtualenv environment by issuing a command of one of the following formats:

```bash
 $ virtualenv --system-site-packages -p python3 targetDirectory 
```

where targetDirectory identifies the top of the Virtualenv tree.

4. Activate the Virtualenv environment by issuing one of the following commands:

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

6. Issue the following commands to install TensorFlow and all the packages that TensorFlow requires into the active Virtualenv environment:

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
A model that use dorpout only on the activation layer

2. [SecondModel](SecondModel.py)
A model that use dropout on each layer, and class weights representation (can be of use if classes are not equivalentaly representated).


## References

As mentionned, this work is mainly based on the following blog post : 
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html


## Author

Nicolas Dujardin, nicolas.dujardin@gmail.com

## License

Those scripts are  available under the MIT license. See the LICENSE file for more info.

