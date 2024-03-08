clear all
clc

% CONFIGURATION
dataset_folder = 'dataset/ai4mars-dataset-merged-0.1/msl';
train_folder   = 'images/edr/train';
test_folder    = 'images/edr/test';
ltrain_folder  = 'labels/train';
ltest_folder   = 'labels/test/masked-gold-min2-100agree';

image_size  = [256, 256, 3];
numClasses  = 5;
divideRatio = 0.85; % #images on train, validation is 1-divideRation

% TAKES TRAIN IMAGES AND DIVIDES INTO TRAIN AND VALIDATION
images = dir(fullfile(dataset_folder, train_folder, '*.jpg'));
images = {images.name};
images = cellfun(@(x) x(1:end-4), images, 'UniformOutput', false);    % names without extension
images = images(randperm(length(images)));                            % shuffle images
train  = images(1:round(divideRatio*length(images)));                 % train images
val   = images(round(divideRatio*length(images))+1:end);              % validation images

% DATASTORES
img_preproc = @(img) imresize(cat(3, imread(img), imread(img), imread(img)), image_size(1:2));
imds_train = imageDatastore(strcat(fullfile(dataset_folder, train_folder, train),'.jpg'), "ReadFcn", img_preproc);
imds_val   = imageDatastore(strcat(fullfile(dataset_folder, train_folder, val),'.jpg'), "ReadFcn", img_preproc);

classes = ["soil","bedrock","sand","bigRock","noLabel"];
labelIDs = [0, 1, 2, 3, 255];
label_preproc = @(img) imresize(imread(img), image_size(1:2));
pxds_train = pixelLabelDatastore(strcat(fullfile(dataset_folder, ltrain_folder, train),'.png'), classes, labelIDs, "ReadFcn", label_preproc);
pxds_val   = pixelLabelDatastore(strcat(fullfile(dataset_folder, ltrain_folder, val),'.png'), classes, labelIDs, "ReadFcn", label_preproc);

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