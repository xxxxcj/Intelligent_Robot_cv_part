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

labels = load("labels.mat");
labels2 = load("labels2.mat");

new_label.gTruth.DataSource = [labels.gTruth.DataSource.Source; labels2.gTruth.DataSource.Source];
new_label.gTruth.LabelDefinitions = labels.gTruth.LabelDefinitions;
new_label.gTruth.LabelData = [labels.gTruth.LabelData; labels2.gTruth.LabelData];
gTruth = new_label.gTruth;
save("all_labels.mat", "gTruth")