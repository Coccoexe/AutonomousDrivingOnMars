clear all
clc

% NETWORK
backbone = 'resnet50'; %resnet18, resnet50, mobilenetv2 ,xception ,inceptionresnetv2

% CONFIGURATION
parent_folder  = 'src/supervised/ai4mars_our';
dataset_folder = 'dataset/ai4mars-dataset-merged-0.4-preprocessed-512';
train_folder   = strcat(dataset_folder, '/images/train');
test_folder    = strcat(dataset_folder, '/images/test');
ltrain_folder  = strcat(dataset_folder, '/labels/train');
ltest_folder   = strcat(dataset_folder, '/labels/test');

divideRatio = 0.8; % #images on train, validation is 1-divideRation
epochPerTrain = 200;
learningRate = 1e-3;
batchSize = 16;
optimizer = 'adam';
loss = 'crossentropy';
image_size = [512 512];

classes = ["soil","bedrock","sand","bigRock","noLabel"];
labelIDs = [0, 1, 2, 3, 255];
numClasses = length(labelIDs);

% TAKES TRAIN IMAGES AND DIVIDES INTO TRAIN AND VALIDATION
images = dir(fullfile(train_folder, '*.png'));
images = {images.name};
images = images(randperm(length(images)));                            % shuffle images
train  = images(1:round(divideRatio*length(images)));                 % train images
val   = images(round(divideRatio*length(images))+1:end);              % validation images

% DATASTORES
imds_train = imageDatastore(fullfile(train_folder, train));
imds_val   = imageDatastore(fullfile(train_folder, val));
pxds_train = pixelLabelDatastore(fullfile(ltrain_folder, train), classes, labelIDs);
pxds_val   = pixelLabelDatastore(fullfile(ltrain_folder, val), classes, labelIDs);
train_cds = combine(imds_train,pxds_train);
val_cds   = combine(imds_val,pxds_val);

% LOAD NETWORK
%layers = deeplabv3plus(image_size, numClasses, backbone);
load(fullfile(parent_folder,'resnet50.mat'));

% NETWORK OPTIONS
chechpoint = strcat(parent_folder,"/checkpoint/");
mkdir(chechpoint);
validation_freq = floor(length(train)/batchSize);
options = trainingOptions(optimizer, ...
    'ExecutionEnvironment','gpu',...
    'InitialLearnRate', learningRate, ...
    'MaxEpochs', epochPerTrain, ...
    'MiniBatchSize', batchSize, ...
    'ValidationData', val_cds, ...
    'ValidationFrequency', validation_freq, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'CheckpointPath', chechpoint, ...
    'CheckpointFrequency', 50);

% TRAIN NETWORK
net = trainnet(train_cds, layers, loss, options);

% SAVE TRAINED NETWORK
time = datetime("now", "Format", "yyMMdd-HHmm");
t = string(time);
name = strcat(string(t),'_',optimizer,'_',string(epochPerTrain));
mkdir(strcat(parent_folder,'/trained_networks/', name));
save(strcat(parent_folder, '/trained_networks/', name, '/trainedNN.mat'), 'net');

% TEST NETWORK
imds_test = imageDatastore(fullfile(test_folder, '*.png'));
pxds_test = pixelLabelDatastore(fullfile(ltest_folder, '*.png'), classes, labelIDs);
test_cds = combine(imds_test,pxds_test);
pxdsResults = semanticseg(imds_test, net, 'MiniBatchSize', 8, 'WriteLocation', tempdir, 'Verbose', false);
metrics = evaluateSemanticSegmentation(pxdsResults, pxds_test, 'Verbose', false);
save(strcat(parent_folder,'/trained_networks/', name, '/metrics.mat'), 'metrics');
