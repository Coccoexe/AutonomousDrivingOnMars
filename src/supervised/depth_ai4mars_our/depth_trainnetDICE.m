clear all
clc

addpath('src/supervised/loss/');

% NETWORK
backbone = 'FuseResNet';

% CONFIGURATIOn
base_network   = 'src/supervised/base_network';
parent_folder = 'src/supervised/depth_ai4mars_our';
rgb_dataset_folder = 'dataset/ai4mars-dataset-merged-0.4-preprocessed-512';
depth_dataset_folder = 'dataset/ai4mars-depth-512';
rgbFolder        = strcat(rgb_dataset_folder,  '/images/train/');
depthFolder      = strcat(depth_dataset_folder,'/images/train/');
labelFolder      = strcat(rgb_dataset_folder,  '/labels/train/');
rgbFolder_test   = strcat(rgb_dataset_folder,  '/images/test/');
depthFolder_test = strcat(depth_dataset_folder,'/images/test/');
labelFolder_test = strcat(rgb_dataset_folder,  '/labels/test/');

divideRatio = 0.8; % #images on train, validation is 1-divideRation
epochPerTrain = 400;
learningRate = 1e-4;
batchSize = 16;
optimizer = 'adam';
loss = @diceLoss;

classes = ["soil","bedrock","sand","bigRock","noLabel"];
labelIDs = [0, 1, 2, 3, 255];
numClasses = length(labelIDs);

% TAKES TRAIN IMAGES AND DIVIDES INTO TRAIN AND VALIDATION
images = dir(fullfile(rgbFolder, '*.png'));
images = {images.name};
images = images(randperm(length(images)));                           
train  = images(1:round(divideRatio*length(images)));                 
val   = images(round(divideRatio*length(images))+1:end);

% DATASTORES
imds_train_rgb = imageDatastore(fullfile(rgbFolder, train));
imds_train_depth = imageDatastore(fullfile(depthFolder, train));
imds_val_rgb = imageDatastore(fullfile(rgbFolder, val));
imds_val_depth = imageDatastore(fullfile(depthFolder, val));
pxds_train = pixelLabelDatastore(fullfile(labelFolder, train), classes, labelIDs);
pxds_val   = pixelLabelDatastore(fullfile(labelFolder, val), classes, labelIDs);
concatenate = @(x,y) cat(3,x,y);
tds_train = transform(imds_train_rgb,imds_train_depth,concatenate);
tds_val = transform(imds_val_rgb,imds_val_depth,concatenate);
ds_train = combine(tds_train,pxds_train);
ds_val = combine(tds_val,pxds_val);

% LOAD NETWORK
net_name = fullfile(base_network,strcat(backbone,'_',int2str(numClasses),'.mat'));
load(net_name);

% NETWORK OPTIONS
chechpoint = strcat("src/supervised/depth/checkpoint/");
mkdir(chechpoint);
validation_freq = floor(length(train)/batchSize);
options = trainingOptions(optimizer, ...
    'ExecutionEnvironment','auto',...
    'InitialLearnRate', learningRate, ...
    'MaxEpochs', epochPerTrain, ...
    'MiniBatchSize', batchSize, ...
    'ValidationData', ds_val, ...
    'ValidationFrequency', validation_freq, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'CheckpointPath', chechpoint, ...
    'CheckpointFrequency', 50);


% TRAIN NET
net = trainnet(ds_train,layers,loss,options);

% SAVE TRAINED NETWORK
time = datetime("now", "Format", "yyMMdd-HHmm");
name = strcat(string(time),'_',backbone);
mkdir(strcat(parent_folder,'/trained_networks/', name));
save(strcat(parent_folder,'/trained_networks/', name, '/config.mat'),'backbone',"batchSize","depth_dataset_folder","divideRatio","epochPerTrain","learningRate","loss","optimizer");
save(strcat(parent_folder, '/trained_networks/', name, '/trainedNN.mat'), 'net');

% TEST NETWORK
imds_test_rgb = imageDatastore(fullfile(rgbFolder_test, '*.png'));
ims_test_depth = imageDatastore(fullfile(depthFolder_test, '*.png'));
pxds_test = pixelLabelDatastore(fullfile(labelFolder_test, '*.png'), classes, labelIDs);
concatenate = @(x,y) cat(3,x,y);
tds_test = transform(imds_test_rgb,ims_test_depth,concatenate);
test_cds = combine(tds_test,pxds_test);
pxdsResults = semanticseg(tds_test, net, 'MiniBatchSize', batchSize, 'WriteLocation', tempdir, 'Verbose', false, 'Classes', classes);
metrics = evaluateSemanticSegmentation(pxdsResults, pxds_test, 'Verbose', false);
save(strcat(parent_folder,'/trained_networks/', name, '/metrics.mat'), 'metrics');

