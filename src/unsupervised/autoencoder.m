% INITIALIZATION - Folders & paths
clear all
clc
train_folder = 'dataset/S5Mars-preprocessed-256/images/train';   % dataset folder
imgS = [256 256];               % image size

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
        convolution2dLayer([3 3], 32, 'Padding', 'same')
        batchNormalizationLayer()
        leakyReluLayer()
        convolution2dLayer([3 3], 64, 'Padding', 'same')
        batchNormalizationLayer()
        leakyReluLayer()
        convolution2dLayer([3 3], 3, 'Padding', 'same')
        batchNormalizationLayer()
        leakyReluLayer()
        convolution2dLayer([3 3], 64, 'Padding', 'same')
        batchNormalizationLayer()
        leakyReluLayer()
        convolution2dLayer([3 3], 32, 'Padding', 'same')
        batchNormalizationLayer()
        leakyReluLayer()
        convolution2dLayer([3 3], 3, 'Padding', 'same')
        regressionLayer];
    
    options = trainingOptions('adam', ...
        'MaxEpochs', 30, ...
        'MiniBatchSize', 8, ...
        'InitialLearnRate', 1e-4, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false, ...
        'Plots', 'training-progress');
    
    net = trainNetwork(TRAIN, TRAIN, layers, options);
    save('net_POOL.mat', 'net');

end

encoder = load("unsupervised/net_POOL.mat").net;
a = activations(encoder,DATA{1},'conv_3');
%a = rgb2ycbcr(a);
imwrite(a, 'test_1.png');
imshow(DATA{1})
imshow(a)
