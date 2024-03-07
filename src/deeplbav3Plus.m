clear all
clc

% CONFIGURATION
dataset_folder = 'dataset/ai4mars-dataset-merged-0.1/msl';
input_images = 'images/edr';
label_train = 'labels/train';
label_test = 'labels/test/masked-gold-min2-100agree';

image_size = [1024, 1024, 3];
numClasses = 5;

% Scan the dataset folder
train_files = dir(fullfile(dataset_folder, label_train, '*.png'));
test_files = dir(fullfile(dataset_folder, label_test, '*.png'));

% create datastore

imds = imageDatastore(fullfile(dataset_folder, input_images),'name',train_files(1).name);


imds = imageDatastore(fullfile(dataset_folder, label_train));
classes = ["soil","bedrock","sand","bigRock","noLabel"];
labelIDs = [0, 1, 2, 3, 255];
pxds = pixelLabelDatastore(fullfile(dataset_folder, label_train), classes, labelIDs);




% import network
layers = deeplabv3plusLayers(image_size, numClasses, 'resnet18');

% train network
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 8, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress');

%cds = combine(imds,pxds);
%net = trainNetwork(cds, layers, options);
