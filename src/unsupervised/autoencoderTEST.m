% INITIALIZATION - Folders & paths
clear all
clc
train_folder = 'dataset/S5Mars-preprocessed-1200/images/train';   % dataset folder
imgS = [1200 1200];               % image size

divideRatio = 0.8; % #images on train, validation is 1-divideRation
images = dir(fullfile(train_folder, '*.jpg'));
images = {images.name};
images = images(randperm(length(images)));                            % shuffle images
train  = images(1:round(divideRatio*length(images)));                 % train images
val   = images(round(divideRatio*length(images))+1:end);              % validation images
imds_train = imageDatastore(fullfile(train_folder, train));

DATA = {};
for K = 1:length(imds_train.Files)
    DATA{end + 1} = readimage(imds_train, K);
end

TRAIN = true;

if TRAIN
    
    % DATASET PREPARATION - shuffle & split
    disp('> DATASET PREPARATION:');
    perm = randperm(length(DATA));
    TRAIN = DATA(perm);
    TRAIN = cat(4, TRAIN{:});
    
    % TRAINING - Convolutional Autoencoder
    disp('> TRAINING:');
    inputSize = [imgS(1) imgS(2) 3];
    layers = [...
        imageInputLayer(inputSize, 'Normalization', 'none')
        convolution2dLayer([3 3], 16, 'Stride', 2, 'Padding', 1)
        batchNormalizationLayer()
        leakyReluLayer()
        convolution2dLayer([3 3], 32, 'Stride', 2, 'Padding', 1)
        batchNormalizationLayer()
        leakyReluLayer()
        convolution2dLayer([3 3], 1, 'Padding', 'same')
        batchNormalizationLayer()
        leakyReluLayer()
        transposedConv2dLayer([3,3], 32, 'Cropping', 'same')
        batchNormalizationLayer()
        leakyReluLayer()
        transposedConv2dLayer([3 3], 16, 'Stride', 2, 'Cropping', 'same')
        batchNormalizationLayer()
        leakyReluLayer()
        transposedConv2dLayer([3 3], 3, 'Stride', 2, 'Cropping', 'same')
        regressionLayer];
    
    options = trainingOptions('adam', ...
        'MaxEpochs', 60, ...
        'MiniBatchSize', 8, ...
        'InitialLearnRate', 1e-4, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false, ...
        'Plots', 'training-progress');
    
    net = trainNetwork(TRAIN, TRAIN, layers, options);
    save('src/unsupervised/trained_autoencoer.mat', 'net');

end

encoder = load("src/unsupervised/trained_autoencoder.mat").net;
for i = 1:100
    a = activations(encoder,DATA{i},'conv_3');
    imwrite(a, fullfile('src/unsupervised/output/',images{i}));
end
