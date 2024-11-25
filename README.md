# Autonomous Driving on Mars

This repository is about the research training for the computer engineer master degree in artificial intelligence and robotics at UNIPD.
In this repository we share the code used during our master thesis, including all the neccessary to replicate the results we obtained.

The code is been used on the DEI CLUSTER, to do so we forced some relatives path so the code may need an adjustment in terms of paths.
There are 3 main directories inside the src, Supervised, Semi-supervised and Dataset.
Datasets includes the test and the analysis but more important the preprocessing we made to train with matlab. Every matlab train functions uses the preprocessed dataset.
Supervised folder contains all training files for all models and many of these have adjustable varibles at the beginning of the file.

## Topic
The main topic is Autonomous Driving on Mars

## Dataset
https://drive.google.com/drive/folders/165JHFpqPEco-kyQy3HRPC9E5pzVjRXOg?usp=sharing

## Usage
The training file is the deeplabvPlus matlab file that divides images in training and validation and trains the networks. 
The are some parameters that can be changed to adjust the training:
- divideRatio: define the train-validation split ration
- epochPerTrain: define the number of max epoch for the train
- learningRate
- batchSize

Once the training is completed the network is saved in the trained_networks folder and the net is tested with test set.
             
```
### Contributors
- Alessio Cocco
- Umberto Salviati

### Supervisor
- Prof. Loris Nanni
