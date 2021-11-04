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
% load('dataset/labels2.mat');
% 
% currentPathDataSource = "E:\Py\znjqr";
% newPathDataSource = "dataset";
% alternativePaths = {[currentPathDataSource newPathDataSource]};
% changeFilePaths(gTruth,alternativePaths)
% 
% save('dataset/labels2.mat');

load('dataset.mat');

currentPathDataSource = "C:\Users\17905\Desktop\1";
newPathDataSource = "dataset";
alternativePaths = {[currentPathDataSource newPathDataSource]};
changeFilePaths(gTruth,alternativePaths);

save('labels.mat', "gTruth");

% labels = load("all_labels.mat");
% labels2 = load("challengetask.mat");
% 
% new_label.gTruth.DataSource = [labels.gTruth.DataSource; labels2.gTruth.DataSource.Source];
% ldc = labelDefinitionCreator(labels2.gTruth.LabelDefinitions)
% new_label.gTruth.LabelDefinitions = labels.gTruth.LabelDefinitions;
% new_label.gTruth.LabelData = [labels.gTruth.LabelData; labels2.gTruth.LabelData];
% gTruth = new_label.gTruth;
% save("all_labels.mat", "gTruth")