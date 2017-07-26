clc; close all;
%%
input_dir = 'C:\Users\215-IV\Desktop\projects\gcs\catagory1\';
input_dir2 = 'C:\Users\215-IV\Desktop\projects\gcs\catagory-1\';
image_dims = [50, 50];
input_image = double(imread([input_dir '2c1.jpeg']));
[rowDim, colDim]=size(input_image);
 
filenames = dir(fullfile(input_dir, '*.jpeg'));
filenames2 = dir(fullfile(input_dir2, '*.jpeg'));

num_images = numel(filenames);
images = [];
for n = 1:num_images
    filename = fullfile(input_dir, filenames(n).name);
    img = imread(filename);
    if n == 1
        images = zeros(prod(image_dims), num_images);
    end
    images(:, n) = img(:);
    labels(n,:) = 1;
end

for n = 1:num_images
    filename = fullfile(input_dir2, filenames2(n).name);
    img = imread(filename);
%     if n == 1
%         images = zeros(prod(image_dims), num_images);
%     end
    images(:, n+num_images) = img(:);
    labels(n+num_images,:) = -1;
end



%% PCA
% 1) Calculate the mean of the input face images
% 2 Subtract the mean from the input images to obtain the mean-shifted images
% 3 Calculate the eigenvectors and eigenvalues of the mean-shifted images
% 4 Order the eigenvectors by their corresponding eigenvalues, in decreasing order
% 5 Retain only the eigenvectors with the largest eigenvalues (the principal components)
% 6 Project the mean-shifted images into the eigenspace using the retained eigenvectors


% steps 1 and 2: find the mean image and the mean-shifted input images
mean_face = mean(images, 2);
shifted_images = images - repmat(mean_face, 1, num_images*2);
 
% steps 3 and 4: calculate the ordered eigenvectors and eigenvalues
[A2, eigVec, eigValue] = pca(images);

cumVar=cumsum(eigValue);
cumVarPercent=cumVar/cumVar(end)*100;
plot(cumVarPercent, '.-');
xlabel('No. of eigenvalues');
ylabel('Cumulated variance percentage (%)');
title('Variance percentage vs. no. of eigenvalues');
fprintf('Saving results into eigenFaceResult.mat...\n');
save eigenFaceResult A2 eigVec cumVarPercent rowDim colDim


load eigenFaceResult.mat	% load A2, eigVec, rowDim, colDim, etc
reducedDim=50;			% Display the first 25 eigenfaces
eigenfaces = reshape(eigVec, rowDim, colDim, size(A2,2));
side=ceil(sqrt(reducedDim));
for i=1:reducedDim
	subplot(side,side,i);
% 	imagesc(eigenfaces(:,:,i)); axis image; colormap(gray);
    imshow(eigenfaces(:,:,i),[]);axis image;
	set(gca, 'xticklabel', ''); set(gca, 'yticklabel', '');
end


%%
DS.input=A2;
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
title('simple PCA')



%%
 
% % step 5: only retain the top 'num_eigenfaces' eigenvectors (i.e. the principal components)
% num_eigenfaces = 25;
% evectors = evectors(:, 1:num_eigenfaces);
%  
% % step 6: project the images into the subspace to generate the feature vectors
% features = shifted_images *  evectors' ;
% 
% 
% %%
% 
% input_image = double(imread([input_dir '2c1.jpeg']));
% feature_vec = (input_image(:) - mean_face) * score ;
% similarity_score = arrayfun(@(n) 1 / (1 + norm(features(:,n) - feature_vec)), 1:num_images);
%  
% % find the image with the highest similarity
% [match_score, match_ix] = max(similarity_score);
%  
% % display the result
% figure, imshow([input_image reshape(images(:,match_ix), image_dims)]);
% title(sprintf('matches %s, score %f', filenames(match_ix).name, match_score));
% 
% 
% % % display the eigenvectors
% % figure;
% % for n = 1:num_eigenfaces
% %     subplot(2, ceil(num_eigenfaces/2), n);
% %     evector = reshape(evectors(:,n), image_dims);
% %     imshow(evector);
% % end