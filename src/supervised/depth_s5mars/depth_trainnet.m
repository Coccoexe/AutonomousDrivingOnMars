clear all
clc

backbone = 'FuseResNet';

% CONFIGURATIOn
base_network   = 'src/supervised/base_network';
parent_folder = 'src/supervised/depth_s5mars';
rgb_dataset_folder = 'dataset/S5Mars-preprocessed-512';
depth_dataset_folder = 'dataset/S5Mars-depth-512';
rgbFolder        = strcat(rgb_dataset_folder,  '/images/train/');
depthFolder      = strcat(depth_dataset_folder,'/images/train/');
labelFolder      = strcat(rgb_dataset_folder,  '/labels/train/');
rgbFolder_test   = strcat(rgb_dataset_folder,  '/images/test/');
depthFolder_test = strcat(depth_dataset_folder,'/images/test/');
labelFolder_test = strcat(rgb_dataset_folder,  '/labels/test/');

divideRatio = 0.8; % #images on train, validation is 1-divideRation
epochPerTrain = 200;
learningRate = 1e-3;
batchSize = 16;
optimizer = 'adam';
loss = 'crossentropy';

classes = ["null","sky","ridge","soil","sand","bedrock","rock","rover","trace","hole"];
labelIDs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
numClasses = length(labelIDs);

% TAKES TRAIN IMAGES AND DIVIDES INTO TRAIN AND VALIDATION
rgbImages = dir(fullfile(rgbFolder, '*.png'));
images = {rgbImages.name};
images = images(randperm(length(images)));                           
train  = images(1:round(divideRatio*length(images)));                 
val   = images(round(divideRatio*length(images))+1:end);

% DATASTORES
imds_train_rgb = imageDatastore(fullfile(rgbFolder, train));
imds_train_depth = imageDatastore(fullfile(depthFolder, train));
imds_val_rgb = imageDatastore(fullfile(rgbFolder, val));
imds_val_depth = imageDatastore(fullfile(depthFolder, val));
concatenate = @(x,y) cat(3,x,y);
tds_train = transform(imds_train_rgb,imds_train_depth,concatenate);
tds_val = transform(imds_val_rgb,imds_val_depth,concatenate);
pxds_train = pixelLabelDatastore(fullfile(labelFolder, train), classes, labelIDs);
pxds_val   = pixelLabelDatastore(fullfile(labelFolder, val), classes, labelIDs);
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
t = string(time);
name = strcat(string(t),'_',optimizer,'_',loss,'_',string(epochPerTrain));
mkdir(strcat(parent_folder,'/trained_networks/', name));
save(strcat(parent_folder, '/trained_networks/', name, '/trainedNN.mat'), 'net');

% TEST NETWORK
imds_test_rgb = imageDatastore(fullfile(rgbFolder_test, '*.png'));
ims_test_depth = imageDatastore(fullfile(depthFolder_test, '*.png'));
concatenate = @(x,y) cat(3,x,y);
tds_test = transform(imds_test_rgb,ims_test_depth,concatenate);
pxds_test = pixelLabelDatastore(fullfile(labelFolder_test, '*.png'), classes, labelIDs);
test_cds = combine(tds_test,pxds_test);
pxdsResults = semanticseg(tds_test, net, 'MiniBatchSize', batchSize, 'WriteLocation', tempdir, 'Verbose', false, 'Classes', classes);
metrics = evaluateSemanticSegmentation(pxdsResults, pxds_test, 'Verbose', false);
save(strcat(parent_folder,'/trained_networks/', name, '/metrics.mat'), 'metrics');
