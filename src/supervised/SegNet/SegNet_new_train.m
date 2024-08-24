clear all


% CONFIGURATION
dataset_folder = 'dataset/S5Mars-preprocessed-512';
train_folder   = strcat(dataset_folder, '/images/train');
test_folder    = strcat(dataset_folder, '/images/test');
ltrain_folder  = strcat(dataset_folder, '/labels/train');
ltest_folder   = strcat(dataset_folder, '/labels/test');
net_folder     = 'src/training/networks/trained_netrowks/resnet18';


image_size  = [512, 512, 3];
numClasses  = 10;
divideRatio = 0.8; % #images on train, validation is 1-divideRation
epochPerTrain = 1000;
learningRate = 1e-3;
batchSize = 16;
optimizer = 'sgdm';
loss = 'crossentropy';

% TAKES TRAIN IMAGES AND DIVIDES INTO TRAIN AND VALIDATION
images = dir(fullfile(train_folder, '*.png'));
images = {images.name};
%images = images(end:-1:1);
images = images(randperm(length(images)));                            % shuffle images
train  = images(1:round(divideRatio*length(images)));                 % train images
val   = images(round(divideRatio*length(images))+1:end);              % validation images               

% DATASTORES
imds_train = imageDatastore(fullfile(train_folder, train));
imds_val   = imageDatastore(fullfile(train_folder, val));

classes = ["null","sky","ridge","soil","sand","bedrock","rock","rover","trace","hole"];
labelIDs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
pxds_train = pixelLabelDatastore(fullfile(ltrain_folder, train), classes, labelIDs);
pxds_val   = pixelLabelDatastore(fullfile(ltrain_folder, val), classes, labelIDs);

train_cds = combine(imds_train,pxds_train);
val_cds   = combine(imds_val,pxds_val);

validation_freq = floor(length(train)/batchSize);

time = datetime("now", "Format", "yyMMdd-HHmm");
t = string(time);
name = strcat(t,'_',optimizer,'_',loss,'_',int2str(epochPerTrain));

chechpoint = strcat(net_folder,"/checkpoint",name,'/');
mkdir(chechpoint);

% train network
options = trainingOptions(optimizer, ...
    'ExecutionEnvironment', 'auto', ...
    'InitialLearnRate', learningRate, ...
    'MaxEpochs', epochPerTrain, ...
    'MiniBatchSize', batchSize, ...
    'ValidationData', val_cds, ...
    'ValidationFrequency', validation_freq, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'CheckpointPath', chechpoint, ...
    'CheckpointFrequency', 100);

numClasses = size(labelIDs, 2);
encoderDepth = 9;
%layer = segnetLayers(image_size,numClasses,encoderDepth);
%save('src/supervised/SegNet/segnet_d9_nopretrain.mat', 'layer');
loadedData = load('src/supervised/SegNet/segnet_d9_nopretrain.mat');
layer = loadedData.layer;
layer = removeLayers(layer,'pixelLabels');
layers = dlnetwork(layer);


net = trainnet(train_cds,layers,loss,options);

% save network
mkdir(strcat('src/supervised/depth/trained_networks/', name));
save(strcat('src/supervised/depth/trained_networks/', name, '/trainedNN.mat'), 'net');

%save training image
%currentfig = findall(groot, 'Tag', 'NNET_CNN_TRAININGPLOT_UIFIGURE');
%savefig(currentfig,strcat('src/supervised/depth/trained_networks/', name,'/training.fig'));

% test network
imds_test = imageDatastore(fullfile(test_folder, '*.png'));
pxds_test = pixelLabelDatastore(fullfile(ltest_folder, '*.png'), classes, labelIDs);

% test network
test_cds = combine(imds_test,pxds_test);

pxdsResults = semanticseg(test_cds, net, 'MiniBatchSize', batchSize, 'WriteLocation', tempdir, 'Verbose', false, 'Classes', classes);
metrics = evaluateSemanticSegmentation(pxdsResults, pxds_test, 'Verbose', false);
folder_trained = fullfile(pwd,strcat("src/supervised/SegNet/trained_networks",name));
if ~exist(folder_trained, 'dir')
    % If the folder does not exist, create it
    mkdir(folder_trained);
    disp(['Folder created: ', folder_trained]);
else
    disp(['Folder already exists: ', folder_trained]);
end

save(strcat(folder_trained, '/metrics.mat'), 'metrics');

