# cv-project

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## General info
This aim of this project is to classify fonts in images. The fonts used: Alex Brush', 'Open Sans', 'Sansation', 'Titillium Web','Ubuntu Mono'.


This is achieved using CNN, implemented with fine tuning on resnet50. It also uses augmentation, preprocessing and postprocessing for better reslts.
	
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
	
## Running
To run this project, install it locally using pip:

```
$ pip -r requirements_predict.txt
$ ./create_res_report.py
```

## Building the classifier
