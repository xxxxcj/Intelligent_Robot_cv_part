modelPath = "yolov3_v9_mask.mat";
datasetPath = 'C:\Users\10142\MATLAB\Projects\Intelligent_Robot_cv_part\data\testtable_v9_mask.mat';
inputSize = [240 480 3];

yolov3Detector = load(modelPath).yolov3Detector;

% dataset = load(datasetPath);
% bcDataset = [dataset.gTruth.DataSource.Source, dataset.gTruth.LabelData];
% [datasetR, datasetC] = size(dataset.gTruth.LabelData);
% 
% trainingDataTbl = bcDataset;
% imdsTrain = imageDatastore(trainingDataTbl{:,1});
% bldsTrain = boxLabelDatastore(trainingDataTbl(:, 2:end));
% 
% trainingData = combine(imdsTrain, bldsTrain);

trainingData = load(datasetPath).testData;
[datasetR, datasetC] = size(trainingData.UnderlyingDatastores{1,1}.Files);

validateInputData(trainingData);
preprocessedTrainingData = transform(trainingData, @(data)preprocess(yolov3Detector, data));

for i = 1:datasetR
    data = read(preprocessedTrainingData);
    target_box = data{2};
    target_label = data{3};
    
    I = data{1,1};
    
    [bboxes, scores, labels] = detect(yolov3Detector,I);
    
    [Center, Labels] = getObjectCenter(yolov3Detector, I);
    
    I = insertMarker(I, Center, '*');
    I = insertObjectAnnotation(I,'rectangle',target_box, target_label, 'Color', 'red');
    I = insertObjectAnnotation(I,'rectangle',bboxes, labels);

    fileName = ['dataset/result_v9_mask/' int2str(i) '.jpg'];
    imwrite(I, fileName);
end

results = detect(yolov3Detector,preprocessedTrainingData,'MiniBatchSize',8);

[IoUs, Classes] = evaluateIoUandClass(results, preprocessedTrainingData);
% Evaluate the object detector using Average Precision metric.
[ap,recall,precision] = evaluateDetectionPrecision(results,preprocessedTrainingData);