# Autonomous Driving on Mars

This repository is about the research training for the computer engineer master degree in artificial intelligence and robotics at UNIPD

## Topic
The main topic is ...

## Usage
The dataset used can be automatically downloaded with the dataset_preparation python file.

The dasated also needs a preprocessing phase where the size of image is set to 256x256x3 (size needed to the network to work)

The training file is the deeplabvPlus matlab file that divides images in training and validation and trains the networks. 
The are some parameters that can be changed to adjust the training:
- divideRatio: define the train-validation split ration
- epochPerTrain: define the number of max epoch for the train
- learningRate
- batchSize

Once the training is completed the network is saved in the trained_networks folder and the net is tested with test set.

## Code
```
src
├───deeplabv3plus
|            └───trained_networks   # folder of trained networks
|
├───dataset_preparation.py          # HELP file to download dataset
|
├───dataset_preprocessing.py        # matlab to preprocess dataset (resize and 1 -> 3 channel)
|
└───deeplabv3Plus.m                 # matlab to train a deeplabv3+
             
```
### Contributors
- Alessio Cocco
- Umberto Salviati

### Supervisor
- Prof. Loris Nanni