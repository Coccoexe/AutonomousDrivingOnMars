clear all
clc

% CONFIGURATION
dataset_folder = 'dataset/ai4mars-dataset-merged-0.1/msl';
depyhAnything_folder = 'src/DepthAnything/Depth-Anything-main';
train_folder   = strcat(depyhAnything_folder,'/trainDepth');
test_folder    = strcat(depyhAnything_folder, '/testDepth');
ltrain_folder  = strcat(dataset_folder, '/labels/train');
ltest_folder   = strcat(dataset_folder, '/labels/test/masked-gold-min2-100agree');

dataset_name = 'dataset/ai4mars';
image_size  = [256, 256];

% DATASET FOLDERS GENERATION
delete(strcat(dataset_name, '/*'));
mkdir(dataset_name);
mkdir(strcat(dataset_name, '/images'));
mkdir(strcat(dataset_name, '/images/train'));
mkdir(strcat(dataset_name, '/images/test'));
mkdir(strcat(dataset_name, '/labels'));
mkdir(strcat(dataset_name, '/labels/train'));
mkdir(strcat(dataset_name, '/labels/test'));

% LOAD IMAGES
imds_train = imageDatastore(train_folder);
imds_test  = imageDatastore(test_folder);
imdsl_train = imageDatastore(ltrain_folder);
imdsl_test  = imageDatastore(ltest_folder);

% PREPROCESS IMAGES
resize = @(x) imresize(x, image_size);
imds_train.ReadFcn = @(x) resize(imread(x));
imds_test.ReadFcn  = @(x) resize(imread(x));
imdsl_train.ReadFcn = @(x) resize(imread(x));
imdsl_test.ReadFcn  = @(x) resize(imread(x));

% SAVE IMAGES
disp('Saving Train Images...');
for i = 1:length(imds_train.Files)
    img = read(imds_train);
    imwrite(img, strcat(dataset_name, '/images/train/', num2str(i), '.png'));
end
disp('Saving Test Images...');
for i = 1:length(imds_test.Files)
    img = read(imds_test);
    imwrite(img, strcat(dataset_name, '/images/test/', num2str(i), '.png'));
end
disp('Saving Train Labels...');
for i = 1:length(imdsl_train.Files)
    img = read(imdsl_train);
    imwrite(img, strcat(dataset_name, '/labels/train/', num2str(i), '.png'));
end
disp('Saving Test Labels...');
for i = 1:length(imdsl_test.Files)
    img = read(imdsl_test);
    imwrite(img, strcat(dataset_name, '/labels/test/', num2str(i), '.png'));
end
disp('Done!');