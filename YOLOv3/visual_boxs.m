data = load('C:\Users\10142\MATLAB\Projects\Intelligent_Robot_cv_part\dataset\labels2.mat').gTruth.LabelData;

[r,c] = size(data);
all_boxes = [];
for i = 1:r        % 建立for循环嵌套
    for k = 1:c
        tmp = table2array(data(i,k));
        tmp = cell2mat(tmp);
        if ~isempty(tmp)
            [kr,~] = size(tmp);
            for t = 1:kr
                all_boxes = [all_boxes;tmp(t,3:4) i k];
            end
        end
    end
end

width = all_boxes(:,1);
height = all_boxes(:,2);

figure
plot(width,height, '*')
xlabel('widht')
ylabel('height')
grid on