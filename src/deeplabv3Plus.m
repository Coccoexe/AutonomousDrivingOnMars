clear all
clc

% CONFIGURATION
dataset_folder = 'dataset/ai4mars';
train_folder   = strcat(dataset_folder, '/images/train');
test_folder    = strcat(dataset_folder, '/images/test');
ltrain_folder  = strcat(dataset_folder, '/labels/train');
ltest_folder   = strcat(dataset_folder, '/labels/test');

image_size  = [256, 256, 3];
numClasses  = 5;
divideRatio = 0.8; % #images on train, validation is 1-divideRation

% TAKES TRAIN IMAGES AND DIVIDES INTO TRAIN AND VALIDATION
images = dir(fullfile(train_folder, '*.png'));
images = {images.name};
images = images(randperm(length(images)));                            % shuffle images
train  = images(1:round(divideRatio*length(images)));                 % train images
val   = images(round(divideRatio*length(images))+1:end);              % validation images

% DATASTORES
imds_train = imageDatastore(fullfile(train_folder, train));
imds_val   = imageDatastore(fullfile(train_folder, val));

classes = ["soil","bedrock","sand","bigRock","noLabel"];
labelIDs = [0, 1, 2, 3, 255];
pxds_train = pixelLabelDatastore(fullfile(ltrain_folder, train), classes, labelIDs);
pxds_val   = pixelLabelDatastore(fullfile(ltrain_folder, val), classes, labelIDs);

train_cds = combine(imds_train,pxds_train);
val_cds   = combine(imds_val,pxds_val);

% import network
layers = deeplabv3plusLayers(image_size, numClasses, 'resnet18');

% train network
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 8, ...
    'ValidationData', val_cds, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress');

net = trainNetwork(train_cds, layers, options);

%for i = 1:10
%    read(imds)
%end
%img = read(imds);
%}