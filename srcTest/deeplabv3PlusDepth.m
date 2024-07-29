clear all
clc

% CONFIGURATION
dataset_folder = 'dataset/ai4mars-dataset-merged-0.1/msl';
train_folder   = 'src/DepthAnything/Depth-Anything-main/trainDepth';
test_folder    = 'images/edr/test';
ltrain_folder  = 'labels/train';
ltest_folder   = 'labels/test/masked-gold-min2-100agree';

image_size  = [256, 256, 3];
numClasses  = 5;
divideRatio = 0.80; % #images on train, validation is 1-divideRation

% TAKES TRAIN IMAGES AND DIVIDES INTO TRAIN AND VALIDATION
images = dir(fullfile(train_folder, '*.png'));
images = {images.name};
images = cellfun(@(x) x(1:end-10), images, 'UniformOutput', false);    % names without extension and suffix
images = images(randperm(length(images)));                            % shuffle images
train  = images(1:round(divideRatio*length(images)));                 % train images
val   = images(round(divideRatio*length(images))+1:end);              % validation images

% DATASTORES
% img_preproc = @(img) imresize(imread(img), image_size(1:2));
%imds_train = imageDatastore(strcat(fullfile(train_folder, train),'_depth.png'), "ReadFcn", img_preproc); % adding suffix
%imds_val   = imageDatastore(strcat(fullfile(train_folder, val),'_depth.png'), "ReadFcn", img_preproc);
imds_train = imageDatastore(fullfile(train_folder, train),'_depth.png');
imds_val   = imageDatastore(fullfile(train_folder, val),'_depth.png');

classes = ["soil","bedrock","sand","bigRock","noLabel"];
labelIDs = [0, 1, 2, 3, 255];
label_preproc = @(img) imresize(imread(img), image_size(1:2));
pxds_train = pixelLabelDatastore(strcat(fullfile(dataset_folder, ltrain_folder, train),'.png'), classes, labelIDs, "ReadFcn", label_preproc);
pxds_val   = pixelLabelDatastore(strcat(fullfile(dataset_folder, ltrain_folder, val),'.png'), classes, labelIDs, "ReadFcn", label_preproc);

train_cds = combine(imds_train,pxds_train);
val_cds   = combine(imds_val,pxds_val);

validation_freq = floor(length(train)/8);

% network checkpoint
if ~exist(net_folder, 'dir')
    mkdir(net_folder);
end
% load network if exists
if exist(strcat(net_folder, '/deeplabv3plus_resnet18.mat'), 'file')
    load(strcat(net_folder, '/deeplabv3plus_resnet18.mat'),'net');
else
    % import network
    layers = deeplabv3plusLayers(image_size, numClasses, 'resnet18');
end

% train network
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', epochPerTrain, ...
    'MiniBatchSize', 8, ...
    'ValidationData', val_cds, ...
    'ValidationFrequency', validation_freq, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress');

net = trainNetwork(train_cds, layers, options);

% save network
save(strcat(net_folder, '/deeplabv3plus_resnet18.mat'), 'net');
%for i = 1:10
%    read(imds)
%end
%img = read(imds);
%}