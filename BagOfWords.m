data = load('C:\Users\10142\MATLAB\Projects\Intelligent_Robot_cv_part\dataset\labels.mat');
Dataset = [data.gTruth.DataSource.Source, data.gTruth.LabelData];

% rng(0);
shuffledIndices = randperm(height(Dataset));
idx = floor(0.6 * length(shuffledIndices));
trainingDataTbl = Dataset(shuffledIndices(1:idx), :);
testDataTbl = Dataset(shuffledIndices(idx+1:end), :);

imdsTrain = imageDatastore(trainingDataTbl{:,1});
imdsTest = imageDatastore(testDataTbl{:,1});

bldsTrain = boxLabelDatastore(trainingDataTbl(:, 2:end));
bldsTest = boxLabelDatastore(testDataTbl(:, 2:end));

trainingData = combine(imdsTrain, bldsTrain);
testData = combine(imdsTest, bldsTest);

data = read(testData);
points = detectORBFeatures(rgb2gray(data{1,1}));
[features,validPoints] = extractFeatures(rgb2gray(data{1,1}),points);

figure
imshow(data{1,1})
hold on
plot(validPoints,'ShowScale',false)
hold off

all_features = [];
for i = 1:idx
    data = read(trainingData);
    I = rgb2gray(data{1,1});
    points = detectORBFeatures(I);
    [features,~] = extractFeatures(I,points);
    all_features = [all_features; features.Features];
end


all_features_double = im2double(all_features)*255;
[idx,C] = kmeans(all_features_double, 5);

data = read(testData);
points = detectORBFeatures(rgb2gray(data{1,1}));
[features,validPoints] = extractFeatures(rgb2gray(data{1,1}),points);

[~,idx_test] = pdist2(C,im2double(features.Features)*255,'euclidean','Smallest',1);
locations = im2double(validPoints.Location);

i_locations = locations(25,:);

figure
imshow(data{1,1})
hold on
for i = 1: length(idx_test)
    switch idx_test(i)
        case 1
            plot(locations(i,1), locations(i,2),  '*')
        case 2
            plot(locations(i,1), locations(i,2),  '+')
        case 3
            plot(locations(i,1), locations(i,2), 'x')
        case 4
            plot(locations(i,1), locations(i,2),  '.')
        case 5
            plot(locations(i,1), locations(i,2), 'o')
    end
end
hold off