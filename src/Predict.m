clear all
clc

input_folder = 'dataset/ai4marsNEW/images/test';
output_folder = 'predictions';
net_folder     = 'src/deeplabv3plus/trained_networks/';

net = load(fullfile(net_folder, 'deeplabv3plus_resnet18_240314-1859.mat'),'net');
net = net.net;

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% predict
X = imageDatastore(input_folder);
YPred = semanticseg(X, net, 'WriteLocation', output_folder);