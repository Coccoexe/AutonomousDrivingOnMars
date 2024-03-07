clear all
clc

% CONFIGURATION
dataset_folder = 'dataset/ai4mars-dataset-merged-0.1/msl';
images_train = 'images/edr/train';
images_test = 'images/edr/test';
label_train = 'labels/train';
label_test = 'labels/test/masked-gold-min2-100agree';

image_size = [255, 255, 3];
numClasses = 5;

% create datastore
transform = @(img) imresize(cat(3, imread(img), imread(img), imread(img)), image_size(1:2));
imds = imageDatastore(fullfile(dataset_folder, images_train), "ReadFcn", transform);
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

cds = combine(imds,pxds);
net = trainNetwork(cds, layers, options);

%for i = 1:10
%    read(imds)
%end
%img = read(imds);