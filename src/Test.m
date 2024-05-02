clear all
clc

test_images = 'dataset\ai4mars_preprocessed_ai4mars_ORIGINAL/images/test';
test_labels = 'dataset\ai4mars_preprocessed_ai4mars_ORIGINAL/labels/test';

net_folder     = 'src/deeplabv3plus/trained_networks/resnet18';
trained_net = 'deeplabv3plus_resnet18_240501-1607.mat';
output_folder = fullfile('predictions', trained_net(1:end-4));

net = load(fullfile(net_folder, trained_net),'net');
net = net.net;

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

classes = ["soil","bedrock","sand","bigRock","noLabel"];
labelIDs = [0, 1, 2, 3, 255];

% predict
imds_test = imageDatastore(fullfile(test_images, '*.png'));
pxds_test = pixelLabelDatastore(fullfile(test_labels, '*.png'), classes, labelIDs);
test_cds = combine(imds_test,pxds_test);
pxdsResults = semanticseg(imds_test, net, 'MiniBatchSize', 8, 'WriteLocation', output_folder);
metrics = evaluateSemanticSegmentation(pxdsResults, pxds_test, 'Verbose', false);

% save metrics
save(fullfile(output_folder, 'metrics.mat'), 'metrics');
