% download pretrained model
doTraining = 1;
if ~doTraining && ~exist('yolov2ResNet50VehicleExample_19b.mat','file')    
    disp('Downloading pretrained detector (98 MB)...');
    pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/yolov2ResNet50VehicleExample_19b.mat';
    websave('yolov2ResNet50VehicleExample_19b.mat',pretrainedURL);
end

% load dataset
data = load('C:\Users\10142\MATLAB\Projects\Intelligent_Robot_cv_part\dataset\labels2.mat');
Dataset = [data.gTruth.DataSource.Source, data.gTruth.LabelData];

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

inputSize = [270 480 3];

numClasses = width(Dataset)-1;

trainingDataForEstimation = transform(trainingData,@(data)preprocessData(data,inputSize));
numAnchors = 4;
[anchorBoxes, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors);

net = mobilenetv2();
lgraph = layerGraph(net);

imgLayer = imageInputLayer(inputSize,"Name","input_1");
lgraph = replaceLayer(lgraph,"input_1",imgLayer);

featureExtractionLayer = "block_12_add";

filterSize = [3 3];
numFilters = 96;

detectionLayers = [
    convolution2dLayer(filterSize,numFilters,"Name","yolov2Conv1","Padding", "same", "WeightsInitializer",@(sz)randn(sz)*0.01)
    batchNormalizationLayer("Name","yolov2Batch1")
    reluLayer("Name","yolov2Relu1")
    convolution2dLayer(filterSize,numFilters,"Name","yolov2Conv2","Padding", "same", "WeightsInitializer",@(sz)randn(sz)*0.01)
    batchNormalizationLayer("Name","yolov2Batch2")
    reluLayer("Name","yolov2Relu2")
    ];

numPredictionsPerAnchor = 5;

numFiltersInLastConvLayer = numAnchors*(numClasses+numPredictionsPerAnchor);

detectionLayers = [
    detectionLayers
    convolution2dLayer(1,numFiltersInLastConvLayer,"Name","yolov2ClassConv",...
    "WeightsInitializer", @(sz)randn(sz)*0.01)
    yolov2TransformLayer(numAnchors,"Name","yolov2Transform")
    yolov2OutputLayer(anchorBoxes,"Name","yolov2OutputLayer")
    ];

lgraph = addLayers(lgraph,detectionLayers);
lgraph = connectLayers(lgraph,featureExtractionLayer,"yolov2Conv1");

augmentedTrainingData = transform(trainingData, @augmentData);

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
figure
imshow(annotatedImage)

options = trainingOptions('adam', ...
        'Verbose',true,...
        'VerboseFrequency',10,...
        'MiniBatchSize',16, ....
        'InitialLearnRate',1e-3, ...
        'MaxEpochs',20, ...
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