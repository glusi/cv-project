# cv-project

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Data](#Data)
* [Running](#Running)
* [Building the classifier](#Building-the-classifier)
* [Output](#Output)
* 
## General info
This aim of this project is to classify fonts in images. 
The available fonts are: 
* Alex Brush
* Open Sans
* Sansation
* Titillium Web
* Ubuntu Mono


This is achieved using CNN, implemented with fine tuning on Resnet50. It also uses augmentation, preprocessing and postprocessing for better reslts.
	
## Technologies
The project is created with:
* h5py==3.7.0
* keras==2.11.0
* matplotlib==3.5.2
* numpy==1.21.5
* opencv_python==4.7.0.68
* requests==2.28.1
* scikit_learn==1.2.1
* tensorflow==2.11.0
* tensorflow_intel==2.11.0
* tqdm==4.64.1

## Data
The training data can be found in the following [link](https://drive.google.com/drive/folders/1jzHYpTwywUYA53nMGHVROSuVO14hEueq?usp=sharing). 
The test set can be flound un the following [link](https://drive.google.com/drive/folders/1hmPI7KaWcv-OLwJEQvMNjbOu9IhU_7CR?usp=sharing).
	
## Running
To run this project, install it locally using pip:
1. Download the model from the link.[TODO]
2. Install and run:

```
$ pip -r requirements_predict.txt
$ ./create_res_report.py
```

## Output
The project creates a file named test_labels.csv.

This file contains all the predictions for each character for each image, in the given dataset.

## Building the classifier
To run this project, install it locally using pip:
1. install dependencies:
```
$ pip -r requirements_build.txt
```
2. run the notebook ```font-recognition.ipynb```
