doTraining = 1;
loadPretrainedModel = 1;

networkInputSize = [270 480 3];
datasetPath = 'C:\Users\10142\MATLAB\Projects\Intelligent_Robot_cv_part\dataset\labels2.mat';

% parameters
numEpochs = 40;
miniBatchSize = 64;
learningRate = 0.001;
warmupPeriod = 1000;
l2Regularization = 0.0005;
penaltyThreshold = 0.5;
velocity = [];

numAnchors = 4;
midAnchors = 2;

mask = im2double(imread("mask1.jpg"));

if canUseParallelPool
   dispatchInBackground = true;
else
   dispatchInBackground = false;
end

if ~doTraining || loadPretrainedModel
    preTrainedDetector = load('yolov3_v9_mask.mat').yolov3Detector; 
end

data = load(datasetPath);
bottleCanDataset = [data.gTruth.DataSource.Source, data.gTruth.LabelData];

for i = 1:10
    bottleCanDataset = [bottleCanDataset; bottleCanDataset(289:336,:)];
end

if doTraining
    % rng(0);
    shuffledIndices = randperm(height(bottleCanDataset));
    idx = floor(0.8 * length(shuffledIndices));
    trainingDataTbl = bottleCanDataset(shuffledIndices(1:idx), :);
    testDataTbl = bottleCanDataset(shuffledIndices(idx+1:end), :);

    imdsTrain = imageDatastore(trainingDataTbl{:,1});
    imdsTest = imageDatastore(testDataTbl{:,1});

    bldsTrain = boxLabelDatastore(trainingDataTbl(:, 2:end));
    bldsTest = boxLabelDatastore(testDataTbl(:, 2:end));

    trainingData = combine(imdsTrain, bldsTrain);
    testData = combine(imdsTest, bldsTest);

    validateInputData(trainingData);
    validateInputData(testData);

    augmentedTrainingData = transform(trainingData, @augmentData);

    % Visualize the augmented images.
    augmentedData = cell(4,1);
    for k = 1:4
        data = read(augmentedTrainingData);
        augmentedData{k} = insertShape(data{1,1}, 'Rectangle', data{1,2}, 'LineWidth',5);
        reset(augmentedTrainingData);
    end
    figure
    montage(augmentedData, 'BorderSize', 10);

    % rng(0)
    trainingDataForEstimation = transform(trainingData, @(data)preprocessData(data, networkInputSize));
    [anchors, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors)

    area = anchors(:, 1).*anchors(:, 2);
    [~, idx] = sort(area, 'descend');
    anchors = anchors(idx, :);
    anchorBoxes = {anchors(1:midAnchors,:)
        anchors(midAnchors+1:end,:)
        };

    baseNetwork = squeezenet;
    classNames = trainingDataTbl.Properties.VariableNames(2:end);

    if loadPretrainedModel
        yolov3Detector = preTrainedDetector;
    else
        yolov3Detector = yolov3ObjectDetector(baseNetwork, classNames, anchorBoxes, ...
            'DetectionNetworkSource', {'fire9-concat', 'fire5-concat'},...
            'InputSize',networkInputSize(1:2));
    end

    preprocessedTrainingData = transform(augmentedTrainingData, @(data)wrapPreprocess(yolov3Detector, data, mask));
    preprocessedTestData = transform(testData, @(data)preprocess(yolov3Detector, data));

    % data = read(preprocessedTrainingData);
    % 
    % I = data{1,1};
    % bbox = data{1,2};
    % annotatedImage = insertShape(I, 'Rectangle', bbox);
    % annotatedImage = imresize(annotatedImage,2);
    % figure
    % imshow(annotatedImage)

    reset(preprocessedTrainingData);
    
    mbqTrain = minibatchqueue(preprocessedTrainingData, 2,...
        "MiniBatchSize", miniBatchSize,...
        "MiniBatchFcn", @(images, boxes, labels) createBatchData(images, boxes, labels, classNames), ...
        "MiniBatchFormat", ["SSCB", ""],...
        "DispatchInBackground", dispatchInBackground,...
        "OutputCast", ["", "double"]);
    
    mbqTest = minibatchqueue(preprocessedTestData, 2,...
        "MiniBatchSize", miniBatchSize,...
        "MiniBatchFcn", @(images, boxes, labels) createBatchData(images, boxes, labels, classNames), ...
        "MiniBatchFormat", ["SSCB", ""],...
        "DispatchInBackground", dispatchInBackground,...
        "OutputCast", ["", "double"]);
    
    % Create subplots for the learning rate and mini-batch loss.
    fig = figure;
    [lossPlotter, learningRatePlotter] = configureTrainingProgressPlotter(fig);
    testFig = figure;
    [testLossPlotter, testLearningRatePlotter] = configureTrainingProgressPlotter(testFig);

    iteration = 0;
    % Custom training loop.
    for epoch = 1:numEpochs
          
        reset(mbqTrain);
        shuffle(mbqTrain);
        
        reset(mbqTest);
        shuffle(mbqTest);
        
        while(hasdata(mbqTrain))
            iteration = iteration + 1;
           
            [XTrain, YTrain] = next(mbqTrain);
            
            % Evaluate the model gradients and loss using dlfeval and the
            % modelGradients function.
            [gradients, state, lossInfo] = dlfeval(@modelGradients, yolov3Detector, XTrain, YTrain, penaltyThreshold);
    
            % Apply L2 regularization.
            gradients = dlupdate(@(g,w) g + l2Regularization*w, gradients, yolov3Detector.Learnables);
    
            % Determine the current learning rate value.
            currentLR = piecewiseLearningRateWithWarmup(iteration, epoch, learningRate, warmupPeriod, numEpochs);
    
            % Update the detector learnable parameters using the SGDM optimizer.
            [yolov3Detector.Learnables, velocity] = sgdmupdate(yolov3Detector.Learnables, gradients, velocity, currentLR);
    
            % Update the state parameters of dlnetwork.
            yolov3Detector.State = state;
              
            % Display progress.
            displayLossInfo(epoch, iteration, currentLR, lossInfo);  
                
            % Update training plot with new points.
            updatePlots(lossPlotter, learningRatePlotter, iteration, currentLR, lossInfo.totalLoss);
            
            if mod(iteration, 20) == 0
                [XTest, YTest] = next(mbqTest);
                [~, ~, testLossInfo] = dlfeval(@modelGradients, yolov3Detector, XTest, YTest, penaltyThreshold);
                displayLossInfo('test', iteration, 'test', testLossInfo); 
                updatePlots(testLossPlotter, testLearningRatePlotter, iteration, currentLR, testLossInfo.totalLoss);
            end
        end   
    end
else
    yolov3Detector = preTrainedDetector;
    testDataTbl = bottleCanDataset;

    imdsTest = imageDatastore(testDataTbl{:,1});

    bldsTest = boxLabelDatastore(testDataTbl(:, 2:end));

    testData = combine(imdsTest, bldsTest);

    validateInputData(testData);
end

results = detect(yolov3Detector,testData,'MiniBatchSize',8);

% Evaluate the object detector using Average Precision metric.
[ap,recall,precision] = evaluateDetectionPrecision(results,testData);
% 
% recall = cell2mat(recall);
% precision = cell2mat(precision);
% 
% % Plot precision-recall curve.
% figure
% plot(recall,precision)
% xlabel('Recall')
% ylabel('Precision')
% grid on
% title(sprintf('Average Precision = %.2f', ap))

% Read the datastore.
data = read(testData);

% Get the image.
I = data{1};

[bboxes,scores,labels] = detect(yolov3Detector,I);

% Display the detections on image.
I = insertObjectAnnotation(I,'rectangle',bboxes,labels);

figure
imshow(I)