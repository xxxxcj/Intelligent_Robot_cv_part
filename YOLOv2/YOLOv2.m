% download pretrained model
doTraining = 1;
if ~doTraining && ~exist('yolov2ResNet50VehicleExample_19b.mat','file')    
    disp('Downloading pretrained detector (98 MB)...');
    pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/yolov2ResNet50VehicleExample_19b.mat';
    websave('yolov2ResNet50VehicleExample_19b.mat',pretrainedURL);
end

% load dataset
data = load('C:\Users\10142\MATLAB\Projects\Intelligent_Robot_cv_part\dataset\all_labels.mat');
Dataset = [data.gTruth.DataSource, data.gTruth.LabelData];

% Display first few rows of the data set.
Dataset(1:4,:)

% rng(0);
shuffledIndices = randperm(height(Dataset));
idx = floor(0.7 * length(shuffledIndices) );

trainingIdx = 1:idx;
trainingDataTbl = Dataset(shuffledIndices(trainingIdx),:);

validationIdx = idx+1 : idx + 1 + floor(0.1 * length(shuffledIndices) );
validationDataTbl = Dataset(shuffledIndices(validationIdx),:);

testIdx = validationIdx(end)+1 : length(shuffledIndices);
testDataTbl = Dataset(shuffledIndices(testIdx),:);

imdsTrain = imageDatastore(trainingDataTbl{:,1});
bldsTrain = boxLabelDatastore(trainingDataTbl(:,2:5));

imdsValidation = imageDatastore(validationDataTbl{:,1});
bldsValidation = boxLabelDatastore(validationDataTbl(:,2:5));

imdsTest = imageDatastore(testDataTbl{:,1});
bldsTest = boxLabelDatastore(testDataTbl(:,2:5));

trainingData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsValidation,bldsValidation);
testData = combine(imdsTest,bldsTest);

data = read(trainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
% figure
% imshow(annotatedImage)

inputSize = [512 1024 3];

numClasses = width(Dataset)-1;

trainingDataForEstimation = transform(trainingData,@(data)preprocessData(data,inputSize));
numAnchors = 2;
[anchorBoxes, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors);

featureExtractionNetwork = resnet50('weight','none');

featureLayer = 'activation_40_relu';

lgraph = yolov2Layers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);

augmentedTrainingData = transform(trainingData,@augmentData);

% Visualize the augmented images.
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},'Rectangle',data{2},'LineWidth',5);
    reset(augmentedTrainingData);
end
% figure
% montage(augmentedData,'BorderSize',10)

preprocessedTrainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
preprocessedValidationData = transform(validationData,@(data)preprocessData(data,inputSize));

data = read(preprocessedTrainingData);

I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
% figure
% imshow(annotatedImage)

options = trainingOptions('adam', ...
        'Verbose',true,...
        'VerboseFrequency',10,...
        'MiniBatchSize',8, ....
        'InitialLearnRate',1e-5, ...
        'MaxEpochs',80, ...
        'CheckpointPath',tempdir, ...
        'ValidationData',preprocessedValidationData);
    
if doTraining       
    % Train the YOLO v2 detector.
    [detector,info] = trainYOLOv2ObjectDetector(preprocessedTrainingData,lgraph,options);
else
    % Load pretrained detector for the example.
    pretrained = load('yolov2ResNet50VehicleExample_19b.mat');
    detector = pretrained.detector;
end

I = imread('dataset\001.jpg');
I = imresize(I,inputSize(1:2));
[bboxes,scores, labels] = detect(detector,I, 'Threshold', 0.1);

I = insertObjectAnnotation(I,'rectangle',bboxes, labels);
figure
imshow(I)

preprocessedTestData = transform(testData,@(data)preprocessData(data,inputSize));

detectionResults = detect(detector, preprocessedTestData);

[ap,recall,precision] = evaluateDetectionPrecision(detectionResults, preprocessedTestData);

figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f',ap))

