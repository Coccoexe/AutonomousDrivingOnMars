clear all
clc

% DATASET PREPROCESSING
DATASET_NAME = 'ai4mars-dataset-merged-0.4';
% CONFIGURATION
dataset_folder = strcat('dataset/',DATASET_NAME);
train_folder   = strcat(dataset_folder, '/images/train');
test_folder    = strcat(dataset_folder, '/images/test');
ltrain_folder  = strcat(dataset_folder, '/labels/train');
ltest_folder   = strcat(dataset_folder, '/labels/test');

image_size  = [513, 513];
dataset_name = strcat('dataset/',DATASET_NAME,'-preprocessed-',num2str(image_size(1)));


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
kernel = strel('square', 20);

resize = @(x) imresize(x, image_size);
duplicate = @(x) cat(3, x, x, x);
imds_train.ReadFcn = @(x) duplicate(resize(imread(x)));
imds_test.ReadFcn  = @(x) duplicate(resize(imread(x)));

imdsl_train.ReadFcn = @(x) resize(imread(x));
imdsl_test.ReadFcn  = @(x) resize(imread(x));

% SAVE IMAGES
disp('Saving Train Images...');
for i = 1:length(imds_train.Files)
    img = read(imds_train);
    imwrite(img, strcat(dataset_name, '/images/train/', imds_train.Files{i}(end-39:end), '.png'));
end
disp('Saving Test Images...');
for i = 1:length(imds_test.Files)
    img = read(imds_test);
    imwrite(img, strcat(dataset_name, '/images/test/', imds_test.Files{i}(end-39:end), '.png'));
end
disp('Saving Train Labels...');
for i = 1:length(imdsl_train.Files)
    img = read(imdsl_train);
    imwrite(img, strcat(dataset_name, '/labels/train/', imdsl_train.Files{i}(end-39:end), '.png'));
end
disp('Saving Test Labels...');
for i = 1:length(imdsl_test.Files)
    img = read(imdsl_test);
    imwrite(img, strcat(dataset_name, '/labels/test/', imdsl_test.Files{i}(end-39:end), '.png'));
end
disp('Done!');