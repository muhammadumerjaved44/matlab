function [dataSet, info, Mean] = featureReductionPCA(dataSet, redudim)
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

X = dataSet.X;
Y = dataSet.Y;

info.pcaRedDim = redudim;
info.DiminssionReducAlgo = 'PCA';
info.pcaInputSzie = size(X,1);
info.pcaInputDim = size(X,2);


[Coeff,Score,Latent,Tsquared,Explained, Mean] = pca(X);

figure;
pareto(Explained)
xlabel('Principal Component')
ylabel('Variance Explained (%)')
title('Principal Components vs Variance')
saveas(gcf,['Noof_Principal_Components' num2str(info.pcaInputSzie)],'jpg')




figure; plot(Score(:,1))
title('varition in first component');

DS.Input=Score(:,1:redudim);
% DS.newLable(1:500,:)= 0;
% DS.newLable(501:1000,:) = 1;

% DS.OutName=classNames;
% DS.Output=Label';
% DS.inputandLables = [DS.Input DS.newLable];

% [nrows, ncols] = size(DS.inputandLables);
% 
% X = DS.inputandLables(:,1:end-1);

dataSet.X = Score(:,1:redudim);
dataSet.Y = Y;
tabulate(dataSet.Y)

figure;
gscatter(Score(:,1),Score(:,2),dataSet.Y,'gr','xo')
xlabel('1st Principal Component')
ylabel('2nd Principal Component')
saveas(gcf,['PrincipalComponentsScatterPlot' num2str(info.pcaInputSzie)],'jpg')


end