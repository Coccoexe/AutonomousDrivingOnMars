clear all
clc

% CONFIGURATION
dataset_folder = 'dataset/ai4mars';
train_folder   = strcat(dataset_folder, '/images/train');
test_folder    = strcat(dataset_folder, '/images/test');
ltrain_folder  = strcat(dataset_folder, '/labels/train');
ltest_folder   = strcat(dataset_folder, '/labels/test');
net_folder     = 'src/deeplabv3plus/trained_networks';


classes = ["soil","bedrock","sand","bigRock","noLabel"];
labelIDs = [0, 1, 2, 3, 255];

% Specify the filename of the network
filename = 'deeplabv3plus_resnet18_240309-1207.mat';

% Load the network
loadedStruct = load(fullfile(net_folder, filename));

% The loaded network is stored in the 'net' field of the loaded structure
net = loadedStruct.net;


% test network
imds_test = imageDatastore(fullfile(test_folder, '*.png'));
pxds_test = pixelLabelDatastore(fullfile(ltest_folder, '*.png'), classes, labelIDs);
test_cds = combine(imds_test,pxds_test);
pxdsResults = semanticseg(imds_test, net, 'MiniBatchSize', 8, 'WriteLocation', tempdir, 'Verbose', false);
metrics = evaluateSemanticSegmentation(pxdsResults, pxds_test, 'Verbose', false);
disp('done');

