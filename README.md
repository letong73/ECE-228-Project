# An Exploration on Improving Signal Modulation Classification with Deep Learning

This is our group project for UCSD ECE 228. It's about Signal Modulation Classification using Deep Learning.


# Project structure description

The project contains different neural network trainings. Each **directory** contains the trainings for a different **data version**. Inside each directory there are the **.ipynb** files which are the **neural network trainings** for the data version specified by the directory where they are. The schema below describes an example of the directory tree structure used in this project.

	.
	├── Data
	│
	├── IQ
	│	│
	│	├── FreeHandV4.ipynb
	│	├── SCRNN.ipynb
	│	└── Freehand_SCRNN.ipynb
	│
	├── IQ-Data_Augmentation
	│	│
	│	├── FreeHandV4_rotation.ipynb
    │       ├── FreeHandV4_rotation_tmaxavg.ipynb 
	│	├── SCRNN_rotation.ipynb 
	│	└── SCRNN_rotation_tmaxavg.ipynb
	

## Directories

The project contains different neural network trainings. Inside the **IQ** directory there are the ones with the **raw dataset**, the other one is using a **data-augmentation of the dataset** before the training. All the directories content is shown in the table below.

| Directory              | Content description                                              |
|------------------------|------------------------------------------------------------------|
| IQ - Data_Augmentation | Raw dataset enlarged using data augmentation tecnique (rotation) |
| IQ                     | Raw dataset                                                      |


## Libraries

We use some libraries to employ in the neural network trainings. A short description of those libraries content is shown in the list below:

- _dataaugmentationlib.py_ enlarging the dataset using data augmentation tecniques
- _datasetlib.py_ reading the dataset
- _evaluationlib.py_ evaluating models after training
- _neural_networks.py_ neural network implementations
- _trainlib.py_ training neural networks
- _traintestsplitlib.py_ split the dataset in training set and test set

# Prerequisites

Follow those steps to get all running.

## Dataset

Dataset can be downloaded from [here](https://www.deepsig.ai/datasets). Extract the dataset in a directory named _data/_. Dataset file should has to be named _RML2016.10a_dict.pkl_.

## Python and pip

You need to have Python (at least 3.9.x) and pip (included with Python) installed on your system.

## Dependencies provisioning

You need to run `pip install -r requirements.txt` to install this project dependencies.

## CUDA

Check if your GPU is CUDA enabled [here](https://developer.nvidia.com/cuda-gpus).

