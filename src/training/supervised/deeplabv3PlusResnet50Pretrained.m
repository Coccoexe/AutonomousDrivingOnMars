clear all
clc

% CONFIGURATION
dataset_folder = 'dataset/ai4mars_preprocessed_NEW_MERGED_OPT';
train_folder   = strcat(dataset_folder, '/images/train');
test_folder    = strcat(dataset_folder, '/images/test');
ltrain_folder  = strcat(dataset_folder, '/labels/train');
ltest_folder   = strcat(dataset_folder, '/labels/test');
net_folder     = 'src/training/networks/trained_netrowks/resnet50';


image_size  = [513, 513, 3];
numClasses  = 5;
divideRatio = 0.8; % #images on train, validation is 1-divideRation
epochPerTrain = 20;
learningRate = 1e-3;
batchSize = 8;

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

validation_freq = floor(length(train)/8);

% train deeplabv3+ with resnet 101
net = load('src/deeplabv3plus/model-Pretrained/deepLabV3Plus-voc.mat');
net = net.net;
layers = layerGraph(net);

filterSize = 1;
numFilters = 5;
stride = [1, 1];
dilationFactor = [1, 1];
padding = [0, 0, 0, 0];  % [top, bottom, left, right]
paddingValue = 0;
weightLearnRateFactor = 11;
weightedL2Factor = 1;
biasL2Factor = 0;
weightInitializer = 'glorot';
biasInitializer = 'zeros';

l = convolution2dLayer(filterSize, numFilters, ...
    'Stride', stride, ...
    'DilationFactor', dilationFactor, ...
    'Padding', padding, ...
    'PaddingValue', paddingValue, ...
    'WeightLearnRateFactor', weightLearnRateFactor, ...
    'WeightL2Factor', weightedL2Factor, ...
    'BiasL2Factor', biasL2Factor, ...
    'WeightsInitializer', weightInitializer, ...
    'BiasInitializer', biasInitializer, ...
    'Name', 'conv1');


layers = replaceLayer(layers, 'node_398', l);
layers = replaceLayer(layers, 'labels',pixelClassificationLayer('Name','labels','Classes',pxds_train.ClassNames));

% train network
options = trainingOptions('sgdm', ...
    'InitialLearnRate', learningRate, ...
    'MaxEpochs', epochPerTrain, ...
    'MiniBatchSize', batchSize, ...
    'ValidationData', val_cds, ...
    'ValidationFrequency', validation_freq, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'CheckpointPath', net_folder, ...
    'CheckpointFrequency', 1);


net = trainNetwork(train_cds, layers, options);



% save network
time = datetime("now", "Format", "yyMMdd-HHmm");
save(strcat(net_folder, '/deepLabV3Plus-voc_pre-trained_', string(time), '.mat'), 'net');

% test network
imds_test = imageDatastore(fullfile(test_folder, '*.png'));
pxds_test = pixelLabelDatastore(fullfile(ltest_folder, '*.png'), classes, labelIDs);
test_cds = combine(imds_test,pxds_test);
pxdsResults = semanticseg(imds_test, net, 'MiniBatchSize', 8, 'WriteLocation', tempdir, 'Verbose', false);
metrics = evaluateSemanticSegmentation(pxdsResults, pxds_test, 'Verbose', false);