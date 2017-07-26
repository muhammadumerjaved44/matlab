function [dataSet, info] = featureReductionPCA(features, redudim)
%% Feature Reduction by using PCA 
% Inputs
% features: input the Feature Set in "Instances x Features"
% redPcaDim = 10 %for selecting the no of componests form score matrix 

% Outputs: 
% reduDataSet: input the Feature Set as reduced datas set in "instances vs reduced Dim"
% info.pcaInfo: hold the information as given below
%                 info.pcaRedDim = redudim;
%                 info.DiminssionReducAlgo = 'PCA';
%                 info.pcaInputSzie = size(features,1);
%                 info.pcaInputDim = size(features,2);

info.pcaRedDim = redudim;
info.DiminssionReducAlgo = 'PCA';
info.pcaInputSzie = size(features,1);
info.pcaInputDim = size(features,2);


[Coeff,Score,Latent,Tsquared,Explained] = pca(features);

figure;
pareto(Explained)
xlabel('Principal Component')
ylabel('Variance Explained (%)')
title('Principal Components vs Variance')

figure;
plot(Score(:,2),Score(:,1),'+')
xlabel('1st Principal Component')
ylabel('2nd Principal Component')

figure; plot(Score(:,1))
title('varition in first component');

DS.Input=Score(:,1:redudim);
% DS.OutName=classNames;
% DS.Output=Label';
DS.newLable(1:500,:)= 1;
DS.newLable(501:1000,:) = 0;
DS.inputandLables = [DS.Input DS.newLable];

[nrows, ncols] = size(DS.inputandLables);

X = DS.inputandLables(:,1:end-1);

dataSet.X = DS.inputandLables(:,1:end-1);
dataSet.Y = DS.inputandLables(:,ncols);
tabulate(dataSet.Y)


end