clc; clear all; close all;
redPcaDim = 10;

%% Load the Data from Directory
% Inputs: set the folder name 
% dataFolder = 'dataSet';
% folderNameC1 = 'catagory1';
% folderNameC2 = 'catagory-1';

% Ouptus:
% features: return the Feature Set in "Instances x Features"
% classLables : returnt the class lables
% classNames : return the unique name for the classes

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


[partitionData, info.holdInfo, testindices]= dataPartitioningHoldout(dataSet,holdoutRatio);
test.X = partitionData.X2input;
test.Y = partitionData.Y2output;
train.X = partitionData.X1input;
train.Y = partitionData.Y1output;



%% Feature Reduction by using PCA for training
% Inputs
% features: input the Feature Set in "Instances x Features"
% redPcaDim = 10 %for selecting the no of componests form score matrix 

% Outputs: 
% reduDataSet: output the Feature Set as reduced datas set in "instances vs reduced Dim"
% info.pcaInfo: hold the information as given below
%                 info.pcaRedDim = redudim;
%                 info.DiminssionReducAlgo = 'PCA';
%                 info.pcaInputSzie = size(features,1);
%                 info.pcaInputDim = size(features,2);

[reduDataTraingSet, info.pcaTrainInfo] = featureReductionPCA(train, redPcaDim);
% mydataSet = [reduDataSet.X reduDataSet.Y];




%%
for i = 1:1

    [partitionDataVlidate, info.validInfo, validindices]= dataPartitioningHoldout(reduDataTraingSet,validationRatio);
    valid.X = partitionDataVlidate.X2input;
    valid.Y = partitionDataVlidate.Y2output;
    train.X = partitionDataVlidate.X1input;
    train.Y = partitionDataVlidate.Y1output;
    classPerformance = classperf(reduDataTraingSet.Y);
%     svmStruct = fitcsvm(train.X,train.Y, 'Standardize',true);
%     figure;
%     cidx = kmeans(train.X,2);
%     silhouette(train.X,cidx);
%     
%     figure;
%     tree = linkage(train.X,'average');
%     dendrogram(tree,0)
%     
%     knnStruct = fitcknn(train.X,train.Y,'Distance','mahalanobis','NumNeighbors',2,'Standardize',true );
    knnStruct = fitcknn(train.X,train.Y, 'NumNeighbors',5,'Standardize',true);
    
    [predictedLabels,score,cost]= predict(knnStruct, valid.X);
%     classperf(CP, train.Y, testidx)
    classperf(classPerformance, predictedLabels, validindices);
    classPerformance
    Accuracy = classPerformance.CorrectRate;
    Accuracy_percnt(i) = Accuracy.*100;
    
    
    %
    Sensitivity(i) = classPerformance.Sensitivity;
    Specificity(i) = classPerformance.Specificity;
    ErrorRate(i) = classPerformance.ErrorRate;
    
end

MAX_ACCURECEY = max(Accuracy_percnt);

if MAX_ACCURECEY > 100
    MAX_ACCURECEY = MAX_ACCURECEY - 1.0;
end

fprintf('accurecey is %d \n', MAX_ACCURECEY);



% T = [struct2table(info.pcaInfo), struct2table(info.validInfo), struct2table(info.holdInfo)];
% writetable(T,'info.xlsx', 'Sheet',1, 'Range','C5')
% 
% ObsVsFeatures = getDiminssion(features, 2);
% FeatureReduction = cell({info.pcaInfo.DiminssionReducAlgo});
% RedcuedFreatureSet = getDiminssion(reduDataSet.X,2);
% Train_Valid_Test_Total = cell({num2str([info.validInfo.trainSzie info.validInfo.validSize info.holdInfo.TestSize ...
%     info.holdInfo.NumObservations])});
% Classification = cell({svmStruct.ModelParameters.Method});
% DateTime = currentDateTime();
% 
% T2= table(ObsVsFeatures,FeatureReduction,RedcuedFreatureSet,...
%     Train_Valid_Test_Total,Classification,DateTime);
% writetable(T2,'info.xlsx', 'Sheet',1, 'Range','C10')

 

%trainMdl = fitcsvm(DS.Input,DS.Output);

% CVSVMModel = fitcsvm(DS.Input,DS.Output,'Holdout',0.2,...
%     'Standardize',true);
% 
hingeLoss = resubLoss(knnStruct,'LossFun','Hinge')
% 
% CompactSVMModel = svmStruct.Trained{1} % Extract trained, compact classifier
% trainkfoldLoss = kfoldLoss(svmStruct)
% 
% % 
% 
% %% Testing Phase
% testInds = test(CVSVMModel.Partition);   % Extract the test indices
% XTest = DS.Input(testInds,:);
% YTest = DS.Output(testInds,:);
% 
% hingeLoss = loss(CompactSVMModel,XTest,YTest,'LossFun','Hinge')
% 
% [label,score] = predict(CompactSVMModel,XTest);
% 
% trueVSpredicted = table(YTest,label,score(:,2),'VariableNames',...
%     {'TrueLabel','PredictedLabel','Score'});
% 
% ConfMat = confusionmat(YTest,label)
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
title('Confusion Matrix of Training Set')
saveas(gcf,'ConfussionMatrixTrain','jpg')



%% plot roc

[x,y,t,AUC] = perfcurve(valid.Y,score(:,1),'0');

figure; plot(x,y);
% axis([XMIN XMAX YMIN YMAX])
axis([-0.1 1.1 -0.1 1.1])
hold on;
plot(min(x),max(y),'r.','MarkerSize',20)
text(min(x)+0.02,max(y)-0.02,['(' num2str(min(x)) ',' num2str(max(y)) ')'])
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC for Classification by SVM of Training Set')
saveas(gcf,'RocPlotTrain','jpg')


%% %%%%%%%%%%%%%%%%%% TESTING the MODEL %%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Feature Reduction by using PCA for training
% Inputs
% features: input the Feature Set in "Instances x Features"
% redPcaDim = 10 %for selecting the no of componests form score matrix 

% Outputs: 
% reduDataSet: output the Feature Set as reduced datas set in "instances vs reduced Dim"
% info.pcaInfo: hold the information as given below
%                 info.pcaRedDim = redudim;
%                 info.DiminssionReducAlgo = 'PCA';
%                 info.pcaInputSzie = size(features,1);
%                 info.pcaInputDim = size(features,2);

[reduDataTestSet, info.pcaTestInfo] = featureReductionPCA(test, redPcaDim);
newTest.X =reduDataTestSet.X;
newTest.Y =reduDataTestSet.Y;
% mydataSet = [reduDataSet.X reduDataSet.Y];


%%

classTestPerformance = classperf(reduDataTestSet.Y);
% svmStruct = fitcsvm(test.X,train.Y, 'Standardize',true);
    
[predictedTestLabels,testScore,testCost]= predict(knnStruct, newTest.X);
%     classperf(CP, train.Y, testidx)
% classperf(classTestPerformance, predictedTestLabels, testindices);
% classTestPerformance
% testAccuracy = classTestPerformance.CorrectRate;
% testAccuracy_percnt = testAccuracy.*100;
%     
%     
% %
% testSensitivity = classTestPerformance.Sensitivity;
% testSpecificity = classTestPerformance.Specificity;
% testErrorRate = classTestPerformance.ErrorRate;

testConfMat = confusionmat(newTest.Y,predictedTestLabels);


[n,p] = size(newTest.X);
isLabels = unique(newTest.Y);
nLabels = numel(isLabels);
[~,grpOOF] = ismember(predictedTestLabels,isLabels); 
oofLabelMat = zeros(nLabels,n); 
idxLinear = sub2ind([nLabels n],grpOOF,(1:n)'); 
oofLabelMat(idxLinear) = 1; % Flags the row corresponding to the class 
[~,grpY] = ismember(newTest.Y,isLabels); 
YMat = zeros(nLabels,n); 
idxLinearY = sub2ind([nLabels n],grpY,(1:n)'); 
YMat(idxLinearY) = 1; 

figure;
plotconfusion(YMat,oofLabelMat);
h = gca;
h.XTickLabel = [isLabels; {' '}];
h.YTickLabel = [isLabels; {' '}];
title('Confusion Matrix on TestData')
saveas(gcf,'ConfussionMatrixTest','jpg')



%% plot roc

[x,y,t,AUC] = perfcurve(newTest.Y,testScore(:,1),'0');

figure; plot(x,y);
% axis([XMIN XMAX YMIN YMAX])
axis([-0.1 1.1 -0.1 1.1])
hold on;
plot(min(x),max(y),'r.','MarkerSize',20)
text(min(x)+0.02,max(y)-0.02,['(' num2str(min(x)) ',' num2str(max(y)) ')'])
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC for Classification by SVM on TestData')
saveas(gcf,'RocPlotTest','jpg')