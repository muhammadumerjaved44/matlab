clc; clear all; close all;
redPcaDim = 10;

%% Load the Data from Directory
% Inputs: set the folder name 
% dataFolder = 'dataSet';

% Ouptus:
% dataSet: return the Feature Set in "Instances x Features"

[dataSet] = loadingData('dataSet');

%% set the partition ratio 

% rng('default') puts the settings of the random number generator used by
%     RAND, RANDI, and RANDN to their default values so that they produce the
%     same random numbers as if you restarted MATLAB.
rng('default');

% Input:
% you just set the partition portion for training, validation and test
% like 60%, 20%, 20% as 60, 20, 20
% getPartitioningRatio(trainRatio, validRatio, testRatio)

% Outputs:
% holdoutRatio: automaticaly give the ratio for holdout data by cvPartition
% function
% validationRatio: automaticaly give the ratio for holdout data by cvPartition
% function

[holdoutRatio, validationRatio] = getPartitioningRatio(60, 20, 20);

%% partitioning the data by Holdout method using cvpartition(Y, 'HoldOut', ratio)
% seperate the data for training and testing

% Input:
% reduDataSet: reduced data set in "instances vs reduced Dim" 
% holdoutRatio: holdout ratio for data partitioning

% Output:
% partitionData
%                 cv: [1x1 cvpartition]
%                 X1input: [800x10 double]
%                 Y1output: [800x1 double]
%                 X2input: [200x10 double]
%                 Y2output: [200x1 double]
% info.holdInfo: 
%                 holdoutRatio: 0.2000
%                 NumObservations: 1000
%                 NumTestSets: 1
%                 TrainSize: 800
%                 TestSize: 200
%                 Type: 'holdout'

[partitionData, info.holdInfo]= dataPartitioningHoldout(reduDataSet,holdoutRatio);

% seperating test and training data
test.X = partitionData.X2input;
test.Y = partitionData.Y2output;
train.X = partitionData.X1input;
train.Y = partitionData.Y1output;

for i = 1:1
    % Input:
    % Give the data set as input for partitioning
    % holdoutRatio: holdout ratio for data partitioning

    % Output:
    % partitionData
    %                 cv: [1x1 cvpartition]
    %                 X1input: [800x10 double]
    %                 Y1output: [800x1 double]
    %                 X2input: [200x10 double]
    %                 Y2output: [200x1 double]
    % info.holdInfo: 
    %                 holdoutRatio: 0.2000
    %                 NumObservations: 1000
    %                 NumTestSets: 1
    %                 TrainSize: 800
    %                 TestSize: 200
    %                 Type: 'holdout'
    % indices:        
    % return the indices of validation set/testing Set form the taining data 
    % and use in calculation of class performance 

    [partitionDataVlidate, info.validInfo, validindices]= dataPartitioningHoldout(train,validationRatio);
    
    % seprating the validation and training set 
    valid.X = partitionDataVlidate.X2input;
    valid.Y = partitionDataVlidate.Y2output;
    newtrain.X = partitionDataVlidate.X1input;
    newtrain.Y = partitionDataVlidate.Y1output;
    
    % to calculate class performance input the all True trainging labeles
    classPerformance = classperf(train.Y);
    
%     MODEL=fitcsvm(X,Y) is an alternative syntax that accepts X as an
%     N-by-P matrix of predictors with one row per observation and one column
%     per predictor. Y is the response and is an array of N class labels. 
    svmStruct = fitcsvm(newtrain.X,newtrain.Y, 'Standardize',true);
    
    [predictedLabels,score,cost]= predict(svmStruct, valid.X);

    %     classperf(CP, train.Y, testidx)
    classperf(classPerformance, predictedLabels, validindices);
    classPerformance
    Accuracy = classPerformance.CorrectRate;
    Accuracy_percnt(i) = Accuracy.*100;
    
    
    %%
    Sensitivity(i) = classPerformance.Sensitivity;
    Specificity(i) = classPerformance.Specificity;
    ErrorRate(i) = classPerformance.ErrorRate;
    
end

MAX_ACCURECEY = max(Accuracy_percnt);

if MAX_ACCURECEY > 100
    MAX_ACCURECEY = MAX_ACCURECEY - 1.0;
end

fprintf('accurecey is %d \n', MAX_ACCURECEY);





writetable(struct2table(info.pcaInfo),'info.xlsx', 'Sheet',1, 'Range','C5')
writetable(struct2table(info.validInfo),'info.xlsx', 'Sheet',1, 'Range','C7')
writetable(struct2table(info.holdInfo),'info.xlsx', 'Sheet',1, 'Range','C9')

ObsVsFeatures = getDiminssion(features, 2);
FeatureReduction = cell({info.pcaInfo.DiminssionReducAlgo});
RedcuedFreatureSet = getDiminssion(reduDataSet.X,2);
Train_Valid_Test_Total = cell({num2str([info.validInfo.trainSzie info.validInfo.validSize info.holdInfo.TestSize ...
    info.holdInfo.NumObservations])});
Classification = cell({svmStruct.ModelParameters.Method});
DateTime = currentDateTime();

T2= table(ObsVsFeatures,FeatureReduction,RedcuedFreatureSet,...
    Train_Valid_Test_Total,Classification,DateTime);
writetable(T2,'info.xlsx', 'Sheet',1, 'Range','C11')

 
hingeLoss = resubLoss(svmStruct,'LossFun','Hinge');

ConfMat = confusionmat(valid.Y,predictedLabels);

% 
% [n,p] = size(XTest);
% isLabels = unique(YTest);
% nLabels = numel(isLabels);
% [~,grpOOF] = ismember(label,isLabels); 
% oofLabelMat = zeros(nLabels,n); 
% idxLinear = sub2ind([nLabels n],grpOOF,(1:n)'); 
% oofLabelMat(idxLinear) = 1; % Flags the row corresponding to the class 
% [~,grpY] = ismember(YTest,isLabels); 
% YMat = zeros(nLabels,n); 
% idxLinearY = sub2ind([nLabels n],grpY,(1:n)'); 
% YMat(idxLinearY) = 1; 
% 
% figure;
% plotconfusion(YMat,oofLabelMat);
% h = gca;
% h.XTickLabel = [isLabels; {' '}];
% h.YTickLabel = [isLabels; {' '}];
% title('simple PCA and SVM')


[n,p] = size(valid.X);
isLabels = unique(valid.Y);
nLabels = numel(isLabels);
[~,grpOOF] = ismember(predictedLabels,isLabels); 
oofLabelMat = zeros(nLabels,n); 
idxLinear = sub2ind([nLabels n],grpOOF,(1:n)'); 
oofLabelMat(idxLinear) = 1; % Flags the row corresponding to the class 
[~,grpY] = ismember(valid.Y,isLabels); 
YMat = zeros(nLabels,n); 
idxLinearY = sub2ind([nLabels n],grpY,(1:n)'); 
YMat(idxLinearY) = 1; 

figure;
plotconfusion(YMat,oofLabelMat);
h = gca;
h.XTickLabel = [isLabels; {' '}];
h.YTickLabel = [isLabels; {' '}];
title('Confusion Matrix')
saveas(gcf,'ConfussionMatrix','jpg')



%% plot roc

[x,y,t,AUC] = perfcurve(valid.Y,score(:,2),'1');

figure; plot(x,y);
% axis([XMIN XMAX YMIN YMAX])
axis([-0.1 1.1 -0.1 1.1])
hold on;
plot(min(x),max(y),'r.','MarkerSize',20)
text(min(x)+0.02,max(y)-0.02,['(' num2str(min(x)) ',' num2str(max(y)) ')'])
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC for Classification by SVM')
saveas(gcf,'RocPlot','jpg')
