yolov3Detector = load("yolov3_v4.mat").yolov3Detector;
inputSize = [227 227 3];

%I = imread('C:\Users\10142\MATLAB\Projects\Intelligent_Robot_cv_part\dataset\010.jpg');
% I = imresize(I,inputSize(1:2));

data = load('C:\Users\10142\MATLAB\Projects\Intelligent_Robot_cv_part\dataset\labels.mat');
vehicleDataset = [data.gTruth.DataSource.Source, data.gTruth.LabelData];

% Add the full path to the local vehicle data folder.
% vehicleDataset.imageFilename = fullfile(pwd, vehicleDataset.imageFilename);

rng(0);
shuffledIndices = randperm(height(vehicleDataset));
idx = floor(0.6 * length(shuffledIndices));
trainingDataTbl = vehicleDataset(shuffledIndices(1:idx), :);
imdsTrain = imageDatastore(trainingDataTbl{:,1});
bldsTrain = boxLabelDatastore(trainingDataTbl(:, 2:end));

trainingData = combine(imdsTrain, bldsTrain);
validateInputData(trainingData);
preprocessedTrainingData = transform(trainingData, @(data)preprocess(yolov3Detector, data));

data = read(preprocessedTrainingData);
I = data{1,1};

[bboxes, scores, labels] = detect(yolov3Detector,I, 'Threshold', 0.1);

I = insertObjectAnnotation(I,'rectangle',bboxes, labels);
figure
imshow(I)