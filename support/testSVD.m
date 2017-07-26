% function testSVD()
close all; clc; clear all;

%% Creating forlders and Path setting
% set folder name to Load image form Class 1 and Class -1 folder
dataFolder = 'dataSet';
folderNameC1 = 'catagory1';
folderNameC2 = 'catagory-1';

% set folder name to save eigen faces
folderNameSVDfaces = 'SVDfaces';
if (~isdir(folderNameSVDfaces))
    mkdir(folderNameSVDfaces)
end

% set folder name to save PCA reconstructed Images
folderNameSVDImages = 'SVDImages';
if (~isdir(folderNameSVDImages))
    mkdir(folderNameSVDImages)
end

%% set path for the image folder



% images form Class 1 and Class -1
myDir1 = [cd '\' dataFolder '\' folderNameC1 '\'];
myDir2 = [cd '\' dataFolder '\' folderNameC2 '\'];
%
outputfolderSVDfaces = [cd '\' folderNameSVDfaces '\'];
addpath(outputfolderSVDfaces);
% Save Recontructed Image to SVDfaces Folder
outputfolderSVDImages = [cd '\' folderNameSVDImages '\'];
addpath(outputfolderSVDImages);

% %%
% ext_img = '*.jpeg';
% a = dir([myDir1 ext_img]);
% nfile = max(size(a)) ; % number of image files
%% set Reduced Dimension for PCA
redudim = 2;

%% folder 1 loading
imagefiles = dir([num2str(myDir1,'%s') '*.jpeg']);      
nfiles = length(imagefiles);    % Number of files found
X=[]
for ii=1:nfiles
   currentfilename = imagefiles(ii).name;
   currentimage =double(imread([myDir1 currentfilename]));
%    images{ii} = currentimage;
%     figure;
%     imshow(currentimage,[])
    temp =reshape(currentimage,[size(currentimage,1)*size(currentimage,2)],1);
    X = [X temp];
    labels(ii,:) = 1;
end

% folder 2 loading
imagefiles2 = dir([num2str(myDir2,'%s') '*.jpeg']);      
nfiles = length(imagefiles2);    % Number of files found
for ii=1:nfiles
   currentfilename2 = imagefiles2(ii).name;
   currentimage2 =double(imread([myDir2 currentfilename2]));
%    images{ii} = currentimage;
%     figure;
%     imshow(currentimage,[])
    temp =reshape(currentimage2,[size(currentimage2,1)*size(currentimage2,2)],1);
    X = [X temp];
    labels(ii+nfiles,:) = -1;
end

%% PCA 
X = X';

[coeff,score,latent,tsquared,explained] = pca(X);

% The first output, coeff, contains the coefficients of the principal components.
% The first three principal component coefficient vectors are:

figure;
pareto(explained)
xlabel('Principal Component')
ylabel('Variance Explained (%)')

figure;
plot(score(:,1),score(:,2),'+')
xlabel('1st Principal Component')
ylabel('2nd Principal Component')


% X is m × n
% U is an m × m unitary matrix (unitary matrices are orthogonal matrices),
% ? or D is a diagonal m × n matrix with non-negative real numbers on the 
%   diagonal diagonal matrix consisting of the set of all eigenvalues of C 
%   along its principal diagonal, and 0 for all other elements
%   
% V is an n × n unitary matrix over K, matrix consisting of the set of all 
%   eigenvectors of C, one eigenvector per column

% W matrix of basis vectors, one vector per column, where each basis vector 
%   is one of the eigenvectors of C, and where the vectors in W are a 
%   sub-set of those in V e.g W = V(:,1:redudim)

% [U D V] = svd(X);
% 
% figure;
% bar(diag(D));
% 
% xhat = U(:,1:redudim)*D(1:redudim,1:redudim)*V(:,1:redudim)';
% %% Eigen Face 
% myEfaces = D(1:redudim,1:redudim)*V(:,1:redudim)';
% 
% for i = 1 : size(myEfaces,1)
% %         eigngcs = [];
%         eigngfaces = reshape(myEfaces(i,:),50,50);
%         eigngfaces = uint8(eigngfaces);
% %         figure;
% %         imshow(eigngfaces)
%         outpath = [outputfolderSVDfaces 'eigen_gcs_faces' num2str(i) '.jpeg'];
%         imwrite(eigngfaces,outpath);
% end
% 
% 
% %% Reconstructed Images using PCA
% for i = 1 : size(xhat,1)
% %         eigngcs = [];
%         eigngcs = reshape(xhat(i,:),50,50);
%         eigngcs = uint8(eigngcs);
% %         figure;
% %         imshow(eigngcs)
%         outpath = [outputfolderSVDImages 'pca_gcs_' num2str(i) '.jpeg'];
%         imwrite(eigngcs,outpath);
% end

%% classification using SVM

% DS.input=;
DS.input=score(:,1:redudim);
DS.outputName=unique(labels);
DS.output=labels;

Mdl = fitcsvm(DS.input,DS.output);

CVMdl = crossval(Mdl);
oosLoss = kfoldLoss(CVMdl);

oofLabel = kfoldPredict(CVMdl);
ConfMat = confusionmat(DS.output,oofLabel);

% Convert the integer label vector to a class-identifier matrix.
[n,p] = size(DS.input);
isLabels = unique(DS.output);
nLabels = numel(isLabels);
[~,grpOOF] = ismember(oofLabel,isLabels); 
oofLabelMat = zeros(nLabels,n); 
idxLinear = sub2ind([nLabels n],grpOOF,(1:n)'); 
oofLabelMat(idxLinear) = 1; % Flags the row corresponding to the class 
[~,grpY] = ismember(DS.output,isLabels); 
YMat = zeros(nLabels,n); 
idxLinearY = sub2ind([nLabels n],grpY,(1:n)'); 
YMat(idxLinearY) = 1; 

figure;
plotconfusion(YMat,oofLabelMat);
h = gca;
h.XTickLabel = [isLabels; {' '}];
h.YTickLabel = [isLabels; {' '}];
title('simple PCA and SVM')

% end