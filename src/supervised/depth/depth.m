clear all

divideRatio = 0.8; % #images on train, validation is 1-divideRation
epochPerTrain = 200;
learningRate = 1e-3;
batchSize = 8;

rgbFolder = 'dataset/S5Mars-preprocessed-512/images/train/';
depthFolder = 'dataset/S5Mars-depth-512/images/train/';
labelFolder = 'dataset/S5Mars-preprocessed-512/labels/train/';

rgbImages = dir(fullfile(rgbFolder, '*.png'));
depthImages = dir(fullfile(depthFolder, '*.png'));

images = {rgbImages.name};
images = images(randperm(length(images)));                           
train  = images(1:round(divideRatio*length(images)));                 
val   = images(round(divideRatio*length(images))+1:end);

% DATASTORES
imds_train_rgb = imageDatastore(fullfile(rgbFolder, train));
ims_train_depth = imageDatastore(fullfile(depthFolder, train));
imds_val_rgb = imageDatastore(fullfile(rgbFolder, val));
imds_val_depth = imageDatastore(fullfile(depthFolder, val));

concatenate = @(x,y) cat(3,x,y);
tds_train = transform(imds_train_rgb,ims_train_depth,concatenate);
tds_val = transform(imds_val_rgb,imds_val_depth,concatenate);

classes = ["null","sky","ridge","soil","sand","bedrock","rock","rover","trace","hole"];
labelIDs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

pxds_train = pixelLabelDatastore(fullfile(labelFolder, train), classes, labelIDs);
pxds_val   = pixelLabelDatastore(fullfile(labelFolder, val), classes, labelIDs);

ds_train = combine(tds_train,pxds_train);
ds_val = combine(tds_val,pxds_val);

chechpoint = strcat("src/supervised/depth/checkpoint/");
mkdir(chechpoint);

validation_freq = floor(length(train)/batchSize);
% train network
options = trainingOptions('adam', ...
    'InitialLearnRate', learningRate, ...
    'MaxEpochs', epochPerTrain, ...
    'MiniBatchSize', batchSize, ...
    'ValidationData', ds_val, ...
    'ValidationFrequency', validation_freq, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'CheckpointPath', chechpoint, ...
    'CheckpointFrequency', 1);

%load net
layers = load('src/supervised/depth/FuseResNet.mat','net').net;
layers = addLayers(layers,pixelClassificationLayer);
layers = connectLayers(layers,'softmax','classoutput');
layers = layerGraph(layers);

%train
net = trainNetwork(ds_train,layers,options);

% save network
time = datetime("now", "Format", "yyMMdd-HHmm");
t = string(time);
mkdir(strcat('src/supervised/depth/trained_networks/', t));
save(strcat('src/supervised/depth/trained_networks/', t, '/trainedNN.mat'), 'net');

%save training image
currentfig = findall(groot, 'Tag', 'NNET_CNN_TRAININGPLOT_UIFIGURE');
savefig(currentfig,strcat('src/supervised/depth/trained_networks/', t,'/training.fig'));

% test network
rgbFolder_test = 'dataset/S5Mars-preprocessed-512/images/test/';
depthFolder_test = 'dataset/S5Mars-depth-512/images/test/';
labelFolder_test = 'dataset/S5Mars-preprocessed-512/labels/test/';
imds_test_rgb = imageDatastore(fullfile(rgbFolder_test, '*.png'));
ims_test_depth = imageDatastore(fullfile(depthFolder_test, '*.png'));

concatenate = @(x,y) cat(3,x,y);
tds_test = transform(imds_test_rgb,ims_test_depth,concatenate);
pxds_test = pixelLabelDatastore(fullfile(labelFolder_test, '*.png'), classes, labelIDs);
test_cds = combine(tds_test,pxds_test);
pxdsResults = semanticseg(tds_test, net, 'MiniBatchSize', 8, 'WriteLocation', tempdir, 'Verbose', false);
metrics = evaluateSemanticSegmentation(pxdsResults, pxds_test, 'Verbose', false);

save(strcat('src/supervised/depth/trained_networks/', t, '/metrics.mat'), 'metrics');
