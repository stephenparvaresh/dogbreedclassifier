## Project Overview

For this project, a Convolutional Neural Network was built to classify dog breeds by user-supplied images.  Given an image of a dog, the algorithm will identify an estimate of the canine’s breed.  If supplied an image of a human, the code will identify the resembling dog breed.  

Two models were created to classify user-supplied images of dogs or humans. The models' performance was measured by the accuracy score, as it is a simple measure of performance, and easily defined by the model. The first CNN model, using ResNet50, pre-defined bottleneck features, and two layers of varying nodes, utilizing sigmoid and relu activations, yielded an accuracy score of 80.26%. The second CNN model, using VGG16, pre-defined bottleneck features, and two layers of varying nodes, again utilizing sigmoid and relu activations, yielded an accuracy score of 77.15%. 

For scoring images of dogs, the ResNet50 model performed better than the VGG16 model, returning more accurate labeling of dogs. However, the ResNet50 model seemed to perform less well when identifiying breeds of dogs when given images of humans, and only returned a few results for vastly different images of humans. This could be an issue of overfitting in the training stage. 

Along with exploring state-of-the-art CNN models for classification, an understanding of the challenges involved in piecing together a series of models designed to perform various tasks in a data processing pipeline is shown.

A link to the Medium article can be found [here](https://medium.com/@stephen.parvaresh/i-am-a-chinese-crested-dog-creating-a-cnn-dog-classifier-8bd284fcf005).

### Libraries Used
- Keras
- OpenCV
- Matplotlib
- Scipy

### Instructions and Additional Files

Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip), which contains test, train, and holdout datasets of dogs.   

Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip), which contain test, train, and holdout datasets of humans. 

Donwload the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset, used in the second model.

(Optional) __If you plan to install TensorFlow with GPU support on your local machine__, follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system.  If you are using an EC2 GPU instance, you can skip this step.

(Optional) **If you are running the project on your local machine (and not using AWS)**, create (and activate) a new environment.

	- __Linux__ (to install with __GPU support__, change `requirements/dog-linux.yml` to `requirements/dog-linux-gpu.yml`): 
	```
	conda env create -f requirements/dog-linux.yml
	source activate dog-project
	```  
	- __Mac__ (to install with __GPU support__, change `requirements/dog-mac.yml` to `requirements/dog-mac-gpu.yml`): 
	```
	conda env create -f requirements/dog-mac.yml
	source activate dog-project
	```  
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/dog-windows.yml` to `requirements/dog-windows-gpu.yml`):  
	```
	conda env create -f requirements/dog-windows.yml
	activate dog-project
	```

(Optional) **If you are running the project on your local machine (and not using AWS)** and Step 6 throws errors, try this __alternative__ step to create your environment.

	- __Linux__ or __Mac__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`): 
	```
	conda create --name dog-project python=3.5
	source activate dog-project
	pip install -r requirements/requirements.txt
	```
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`):  
	```
	conda create --name dog-project python=3.5
	activate dog-project
	pip install -r requirements/requirements.txt
	```
	
(Optional) **If you are using AWS**, install Tensorflow.
```
sudo python3 -m pip install -r requirements/requirements-gpu.txt
```
	
Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
	- __Linux__ or __Mac__: 
		```
		KERAS_BACKEND=tensorflow python -c "from keras import backend"
		```
	- __Windows__: 
		```
		set KERAS_BACKEND=tensorflow
		python -c "from keras import backend"
		```

(Optional) **If you are running the project on your local machine (and not using AWS)**, create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `dog-project` environment. 
```
python -m ipykernel install --user --name dog-project --display-name "dog-project"
```

11. Open the notebook.
```
jupyter notebook dog_app.ipynb
```
