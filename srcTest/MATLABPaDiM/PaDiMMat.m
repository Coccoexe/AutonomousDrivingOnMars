dataDir = fullfile(pwd,"ConcreteCrackDataset");
if ~exist(dataDir,"dir")
    mkdir(dataDir);
end
disp(dataDir);
%%
imdsPositive = imageDatastore(fullfile(dataDir,"Positive"),LabelSource="foldernames");
imdsNegative = imageDatastore(fullfile(dataDir,"Negative"),LabelSource="foldernames");
%%
samplePositive = preview(imdsPositive);
sampleNegative = preview(imdsNegative);
montage({sampleNegative,samplePositive})
title("Road Images Without (Left) and with (Right) Cracks")
%%
numTrainNormal = 250;
numCal = 100;
numTest = 1000;

[imdsTestPos,imdsCalPos] = splitEachLabel(imdsPositive,numTest,numCal);
[imdsTrainNeg,imdsTestNeg,imdsCalNeg] = splitEachLabel(imdsNegative,numTrainNormal,numTest,numCal,"randomized");

trainFiles = imdsTrainNeg.Files;
calibrationFiles = cat(1,imdsCalPos.Files,imdsCalNeg.Files);
testFiles = cat(1,imdsTestPos.Files,imdsTestNeg.Files);

imdsTrain = imageDatastore(trainFiles,LabelSource="foldernames");
imdsCal = imageDatastore(calibrationFiles,LabelSource="foldernames");
imdsTest = imageDatastore(testFiles,LabelSource="foldernames");
%%
addLabelFcn = @(x,info) deal({x,onehotencode(info.Label,1)},info);
tdsTrain = transform(imdsTrain,addLabelFcn,IncludeInfo=true);
tdsCal = transform(imdsCal,addLabelFcn,IncludeInfo=true);
tdsTest = transform(imdsTest,addLabelFcn,IncludeInfo=true);
%%
resizeImageSize = [256 256];
targetImageSize = [224 224];
resizeAndCropImageFcn = @(x,info) deal({resizeAndCropForConcreteAnomalyDetector(x{1},resizeImageSize,targetImageSize),x{2}});
tdsTrain = transform(tdsTrain,resizeAndCropImageFcn);
tdsCal = transform(tdsCal,resizeAndCropImageFcn);
tdsTest = transform(tdsTest,resizeAndCropImageFcn);
%%
minibatchSize = 128;
trainQueue = minibatchqueue(tdsTrain, ...
    PartialMiniBatch="return", ...
    MiniBatchFormat=["SSCB","CB"], ...
    MiniBatchSize=minibatchSize);
%%
net = imagePretrainedNetwork("resnet18");

feature1LayerName = "bn2b_branch2b";
feature2LayerName = "bn3b_branch2b";
feature3LayerName = "bn4b_branch2b";

XTrainFeatures1 = [];
XTrainFeatures2 = [];
XTrainFeatures3 = [];

reset(trainQueue);
shuffle(trainQueue);
idx = 1;
while hasdata(trainQueue)
    [X,T] = next(trainQueue);

    XTrainFeatures1 = cat(4,XTrainFeatures1,predict(net,extractdata(X),Outputs=feature1LayerName));
    XTrainFeatures2 = cat(4,XTrainFeatures2,predict(net,extractdata(X),Outputs=feature2LayerName));
    XTrainFeatures3 = cat(4,XTrainFeatures3,predict(net,extractdata(X),Outputs=feature3LayerName));
    idx = idx+size(X,4);
end
%%
XTrainFeatures1 = gather(XTrainFeatures1);
XTrainFeatures2 = gather(XTrainFeatures2);
XTrainFeatures3 = gather(XTrainFeatures3);
XTrainEmbeddings = concatenateEmbeddings(XTrainFeatures1,XTrainFeatures2,XTrainFeatures3);
%%
whos XTrainEmbeddings
%%
selectedChannels = 100;
totalChannels = 448;
rIdx = randi(totalChannels,[1 selectedChannels]);
XTrainEmbeddings = XTrainEmbeddings(:,:,rIdx,:);
%%
[H, W, C, B] = size(XTrainEmbeddings);
XTrainEmbeddings = reshape(XTrainEmbeddings,[H*W C B]);
%%
means = mean(XTrainEmbeddings,3);
%%
covars = zeros([H*W C C]);
identityMatrix = eye(C);
for idx = 1:H*W
    covars(idx,:,:) = cov(squeeze(XTrainEmbeddings(idx,:,:))') + 0.01* identityMatrix;
end
%%
minibatchSize = 1;
calibrationQueue = minibatchqueue(tdsCal, ...
    MiniBatchFormat=["SSCB","CB"], ...
    MiniBatchSize=minibatchSize, ...
    OutputEnvironment="auto");
%%
maxScoresCal = zeros(tdsCal.numpartitions,1);
minScoresCal = zeros(tdsCal.numpartitions,1);
meanScoresCal = zeros(tdsCal.numpartitions,1);
idx = 1;

while hasdata(calibrationQueue)
    XCal = next(calibrationQueue);
    
    XCalFeatures1 = predict(net,extractdata(XCal),Outputs=feature1LayerName);
    XCalFeatures2 = predict(net,extractdata(XCal),Outputs=feature2LayerName);
    XCalFeatures3 = predict(net,extractdata(XCal),Outputs=feature3LayerName);

    XCalFeatures1 = gather(XCalFeatures1);
    XCalFeatures2 = gather(XCalFeatures2);
    XCalFeatures3 = gather(XCalFeatures3);
    XCalEmbeddings = concatenateEmbeddings(XCalFeatures1,XCalFeatures2,XCalFeatures3);

    XCalEmbeddings = XCalEmbeddings(:,:,rIdx,:);
    [H, W, C, B] = size(XCalEmbeddings);
    XCalEmbeddings = reshape(permute(XCalEmbeddings,[1 2 3 4]),[H*W C B]);

    distances = calculateDistance(XCalEmbeddings,H,W,B,means,covars);

    anomalyScoreMap = createAnomalyScoreMap(distances,H,W,B,targetImageSize);

    % Calculate max, min, and mean values of the anomaly score map
    maxScoresCal(idx:idx+size(XCal,4)-1) = squeeze(max(anomalyScoreMap,[],[1 2 3]));
    minScoresCal(idx:idx+size(XCal,4)-1) = squeeze(min(anomalyScoreMap,[],[1 2 3]));
    meanScoresCal(idx:idx+size(XCal,4)-1) = squeeze(mean(anomalyScoreMap,[1 2 3]));
    
    idx = idx+size(XCal,4);
    clear XCalFeatures1 XCalFeatures2 XCalFeatures3 anomalyScoreMap distances XCalEmbeddings XCal
end
%%
labelsCal = tdsCal.UnderlyingDatastores{1}.Labels ~= "Negative";
%%
maxScore = max(maxScoresCal,[],"all");
minScore = min(minScoresCal,[],"all");

scoresCal =  mat2gray(meanScoresCal, [minScore maxScore]);
%%
maxScore = max(maxScoresCal,[],"all");
minScore = min(minScoresCal,[],"all");

scoresCal =  mat2gray(meanScoresCal, [minScore maxScore]);
%%
[~,edges] = histcounts(scoresCal,20);
hGood = histogram(scoresCal(labelsCal==0),edges);
hold on
hBad = histogram(scoresCal(labelsCal==1),edges);
hold off
legend([hGood,hBad],"Normal (Negative)","Anomaly (Positive)")
xlabel("Mean Anomaly Score");
ylabel("Counts");
%%
[xroc,yroc,troc,auc] = perfcurve(labelsCal,scoresCal,true);
figure
lroc = plot(xroc,yroc);
hold on
lchance = plot([0 1],[0 1],"r--");
hold off
xlabel("False Positive Rate") 
ylabel("True Positive Rate")
title("ROC Curve AUC: "+auc);
legend([lroc,lchance],"ROC curve","Random Chance")
%%
[~,ind] = max(yroc-xroc);
anomalyThreshold = troc(ind)
%%
minibatchSize = 1;
testQueue = minibatchqueue(tdsTest, ...
    MiniBatchFormat=["SSCB","CB"], ...
    MiniBatchSize=minibatchSize, ...
    OutputEnvironment="auto");
%%
idx = 1;

XTestImages = [];
anomalyScoreMapsTest = [];

while hasdata(testQueue)
    XTest = next(testQueue);
    
    XTestFeatures1 = predict(net,extractdata(XTest),Outputs=feature1LayerName);
    XTestFeatures2 = predict(net,extractdata(XTest),Outputs=feature2LayerName);
    XTestFeatures3 = predict(net,extractdata(XTest),Outputs=feature3LayerName);

    XTestFeatures1 = gather(XTestFeatures1);
    XTestFeatures2 = gather(XTestFeatures2);
    XTestFeatures3 = gather(XTestFeatures3);
    XTestEmbeddings = concatenateEmbeddings(XTestFeatures1,XTestFeatures2,XTestFeatures3);
    
    XTestEmbeddings = XTestEmbeddings(:,:,rIdx,:);
    [H, W, C, B] = size(XTestEmbeddings);
    XTestEmbeddings = reshape(XTestEmbeddings,[H*W C B]);

    distances = calculateDistance(XTestEmbeddings,H,W,B,means,covars);

    anomalyScoreMap = createAnomalyScoreMap(distances,H,W,B,targetImageSize);
    XTestImages = cat(4,XTestImages,gather(XTest));
    anomalyScoreMapsTest = cat(4,anomalyScoreMapsTest,gather(anomalyScoreMap));
    
    idx = idx+size(XTest,4);
    clear XTestFeatures1 XTestFeatures2 XTestFeatures3 anomalyScoreMap distances XTestEmbeddings XTest
end
%%
scoresTest = squeeze(mean(anomalyScoreMapsTest,[1 2 3]));
scoresTest = mat2gray(scoresTest,[minScore maxScore]);
%%
predictedLabels = scoresTest > anomalyThreshold;
%%
targetLabels = logical(labelsTest);
M = confusionmat(targetLabels,predictedLabels);
confusionchart(M,["Negative","Positive"])
acc = sum(diag(M)) / sum(M,"all");
title("Accuracy: "+acc);
%%
maxScoresCal = mat2gray(maxScoresCal);
scoreMapRange = [0 prctile(maxScoresCal,80,"all")];
%%
idxTruePositive = find(targetLabels & predictedLabels);
dsTruePositive = subset(tdsTest,idxTruePositive);
dataTruePositive = preview(dsTruePositive);
imgTruePositive = dataTruePositive{1};
imshow(imgTruePositive)
title("True Positive Test Image")
%%
anomalyTestMapsRescaled = mat2gray(anomalyScoreMapsTest,[minScore maxScore]);
scoreMapTruePositive = anomalyTestMapsRescaled(:,:,1,idxTruePositive(1));
%%
imshow(anomalyMapOverlayForConcreteAnomalyDetector(imgTruePositive, ...
    scoreMapTruePositive,ScoreMapRange=scoreMapRange));
title("Heatmap Overlay of True Positive Result")
%%
disp("Mean anomaly score of test image: "+scoresTest(idxTruePositive(1)))
%%
idxTrueNegative = find(~(targetLabels | predictedLabels));
dsTrueNegative = subset(tdsTest,idxTrueNegative);
dataTrueNegative = preview(dsTrueNegative);
imgTrueNegative = dataTrueNegative{1};
imshow(imgTrueNegative)
title("True Negative Test Image")
%%
scoreMapTrueNegative = anomalyTestMapsRescaled(:,:,1,idxTrueNegative(1));
imshow(anomalyMapOverlayForConcreteAnomalyDetector(imgTrueNegative, ...
    scoreMapTrueNegative,ScoreMapRange=scoreMapRange))
title("Heatmap Overlay of True Negative Result")
%%
disp("Mean anomaly score of test image: "+scoresTest(idxTrueNegative(1)))
%%
idxFalsePositive = find(~targetLabels & predictedLabels);
dataFalsePositive = readall(subset(tdsTest,idxFalsePositive));
numelFalsePositive = length(idxFalsePositive);    
numImages = min(numelFalsePositive,3);
if numelFalsePositive>0
    montage(dataFalsePositive(1:numImages,1),Size=[1,numImages],BorderSize=10);
    title("False Positives in Test Set")
end
%%
hmapOverlay = cell(1,numImages);
for idx = 1:numImages
    img = dataFalsePositive{idx,1};
    scoreMapFalsePositive = anomalyTestMapsRescaled(:,:,1,idxFalsePositive(idx));
    hmapOverlay{idx} = anomalyMapOverlayForConcreteAnomalyDetector(img, ...
        scoreMapFalsePositive,ScoreMapRange=scoreMapRange);
end
%%
if numelFalsePositive>0
    montage(hmapOverlay,Size=[1,numImages],BorderSize=10)
    title("Heatmap Overlays of False Positive Results")
end
%%
disp("Mean anomaly scores:"); scoresTest(idxFalsePositive(1:numImages))
%%
idxFalseNegative = find(targetLabels & ~predictedLabels);
dataFalseNegative = readall(subset(tdsTest,idxFalseNegative));
numelFalseNegative = length(idxFalseNegative);
numImages = min(numelFalseNegative,3);
if numelFalseNegative>0
    montage(dataFalseNegative(1:numImages,1),Size=[1,numImages],BorderSize=10);
    title("False Negatives in Test Set")
end
%%
hmapOverlay = cell(1,numImages);
for idx = 1:numImages
    img = dataFalseNegative{idx,1};
    scoreMapFalseNegative = anomalyTestMapsRescaled(:,:,1,idxFalseNegative(idx));
    hmapOverlay{idx} = anomalyMapOverlayForConcreteAnomalyDetector(img, ...
        scoreMapFalseNegative,ScoreMapRange=scoreMapRange);
end
%%
if numelFalseNegative>0
    montage(hmapOverlay,Size=[1,numImages],BorderSize=10)
    title("Heatmap Overlays of False Negative Results")
end
%%
disp("Mean anomaly scores:"); scoresTest(idxFalsePositive(1:numImages))
%%
function XEmbeddings = concatenateEmbeddings(XFeatures1,XFeatures2,XFeatures3)
    XFeatures2Resize = imresize(XFeatures2,2,"nearest");
    XFeatures3Resize = imresize(XFeatures3,4,"nearest");
    XEmbeddings = cat(3,XFeatures1,XFeatures2Resize,XFeatures3Resize);
end
%%
function distances = calculateDistance(XEmbeddings,H,W,B,means,covars)
    distances = zeros([H*W 1 B]);
    for dIdx = 1:H*W
        distances(dIdx,1,:) = pdist2(squeeze(means(dIdx,:)), ...
            squeeze(XEmbeddings(dIdx,:,:)),"mahal",squeeze(covars(dIdx,:,:)));
    end
end
%%
function anomalyScoreMap = createAnomalyScoreMap(distances,H,W,B,targetImageSize)
    anomalyScoreMap = reshape(distances,[H W 1 B]);
    anomalyScoreMap = imresize(anomalyScoreMap,targetImageSize,"bilinear");
    for mIdx = 1:size(anomalyScoreMap,4)
        anomalyScoreMap(:,:,1,mIdx) = imgaussfilt(anomalyScoreMap(:,:,1,mIdx),4,FilterSize=33);
    end
end