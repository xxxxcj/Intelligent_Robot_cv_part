% load('dataset/labels.mat');
% 
% gTruth.DataSource
% 
% currentPathDataSource = "C:\Users\Lenovo\Desktop\1";
% newPathDataSource = "dataset";
% alternativePaths = {[currentPathDataSource newPathDataSource]};
% changeFilePaths(gTruth,alternativePaths);
% 
% save('dataset/labels.mat');
load('dataset/labels.mat');

currentPathDataSource = "E:\Py\znjqr";
newPathDataSource = "dataset";
alternativePaths = {[currentPathDataSource newPathDataSource]};
changeFilePaths(gTruth,alternativePaths)

save('dataset/labels.mat');