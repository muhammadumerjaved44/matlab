clc; clear all; close all;

%%
dataFolder = 'dataSet';
folderNameC1 = 'catagory1';
folderNameC2 = 'catagory-1';

%% set Reduced Dimension for PCA
redudim = 10;

%% Data Partition 

gcsDatabase = imageSet(dataFolder,'recursive');

% [training,test] = partition(gcsDatabase,[1.0 0.0]);


featureCount = 0;
for i=1:size(gcsDatabase,2)
    for j = 1:gcsDatabase(i).Count
        featureCount = featureCount + 1;
        currentimage = double((read(gcsDatabase(i),j)));
        Features(featureCount,:) = reshape(currentimage,size(currentimage,1)*size(currentimage,2),1);
        Label{featureCount} = gcsDatabase(i).Description;    
    end
    classNames{i} = gcsDatabase(i).Description;
end


[Coeff,Score,Latent,Tsquared,Explained] = pca(Features);

figure;
pareto(Explained)
xlabel('Principal Component')
ylabel('Variance Explained (%)')

figure;
plot(Score(:,2),Score(:,1),'+')
xlabel('1st Principal Component')
ylabel('2nd Principal Component')

% varition in first component
figure; plot(Score(:,1))
xlabel('varition in first component');
title('varition in first component');

DS.Input=Score(:,1:redudim);
DS.OutName=classNames;
DS.Output=Label';

% scale (DS.Input , [-1 1])
CVSVMModel = fitcsvm(DS.Input,DS.Output,'HoldOut', 0.2,...
    'Standardize',true);


CompactSVMModel = CVSVMModel.Trained{1} % Extract trained, compact classifier
trainkfoldLoss = kfoldLoss(CVSVMModel)

% 

%% Testing Phase
testInds = test(CVSVMModel.Partition);   % Extract the test indices
XTest = DS.Input(testInds,:);
YTest = DS.Output(testInds,:);

hingeLoss = loss(CompactSVMModel,XTest,YTest,'LossFun', 'hinge')

[label,score] = predict(CompactSVMModel,XTest);

trueVSpredicted = table(YTest,label,score(:,2),'VariableNames',...
    {'TrueLabel','PredictedLabel','Score'});

ConfMat = confusionmat(YTest,label)

[n,p] = size(XTest);
isLabels = unique(YTest);
nLabels = numel(isLabels);
[~,grpOOF] = ismember(label,isLabels); 
oofLabelMat = zeros(nLabels,n); 
idxLinear = sub2ind([nLabels n],grpOOF,(1:n)'); 
oofLabelMat(idxLinear) = 1; % Flags the row corresponding to the class 
[~,grpY] = ismember(YTest,isLabels); 
YMat = zeros(nLabels,n); 
idxLinearY = sub2ind([nLabels n],grpY,(1:n)'); 
YMat(idxLinearY) = 1; 

figure;
plotconfusion(YMat,oofLabelMat);
h = gca;
h.XTickLabel = [isLabels; {' '}];
h.YTickLabel = [isLabels; {' '}];
title('simple PCA and SVM')


%% plot roc
% performance curve for classifier output
[perX,perY,~,AUC] = perfcurve(label,score(:,2),'catagory1');
figure;plot(perX,perY);
xlabel('False positive rate');
ylabel('True positive rate');
title('ROC for Classification by SVM');
