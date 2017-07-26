close all; clc; clear all;

%set path for the image folder

outputfolder = 'C:\Users\215-IV\Desktop\projects\gcs\eigen_gcs\';
redudim = 10; 

myDir1 = 'C:\Users\215-IV\Desktop\projects\gcs\catagory1\';
myDir2 = 'C:\Users\215-IV\Desktop\projects\gcs\catagory-1\'
ext_img = '*.jpeg';
a = dir([myDir1 ext_img]);
nfile = max(size(a)) ; % number of image files
for i=1:nfile
  my_img(i).img = imread([myDir1 num2str(i) 'c1.jpeg']);
  files(i).img = [[myDir1 num2str(i) 'c1.jpeg']];
  labels(i).img = 1;
end

b = dir([myDir2 ext_img]);
nfile = max(size(b)) ; % number of image files
for i=1:nfile
  my_img(i+nfile).img = imread([myDir2 num2str(i) 'c-1.jpeg']);
  files(i+nfile).img = [[myDir2 num2str(i) 'c-1.jpeg']];
  labels(i+nfile).img = -1;
end


% %%
% Afields = fieldnames(a);
% Acell = struct2cell(a);
% sz = size(Acell)
% 
% % Convert to a matrix
% Acell = reshape(Acell, sz(1), []);      % Px(MxN)
% 
% % Make each field a column
% Acell = Acell';                         % (MxN)xP
% 
% % Sort by first field "name"
% Acell = sortrows(Acell, 1)




%% load images form folders

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
end

%%
X = X'
% [U D V] = svd(X);
% 
% figure;
% plot(diag(D));
% 
% xhat = U(:,1:redudim)*D(1:redudim,1:redudim)*V(:,1:redudim)';
% 
% for i = 1 : size(xhat,1)
% %         eigngcs = [];
%         eigngcs = reshape(xhat(i,:),50,50);
%         eigngcs = uint8(eigngcs);
% %         figure;
% %         imshow(eigngcs)
%         outpath = [outputfolder 'eigen_gcs_' num2str(i) '.jpeg'];
%         imwrite(eigngcs,outpath);
% end 

[coeff,score,latent] = pca(X')

newX = score*coeff(1:13,:)';
newX = newX';




for i = 1 : size(newX,1)
%         eigngcs = [];
        eigngcs = reshape(newX(i,:),50,50);
        eigngcs = uint8(eigngcs);
%         figure;
%         imshow(eigngcs)
        outpath = [outputfolder 'eigen_gcs_' num2str(i) '.jpeg'];
        imwrite(eigngcs,outpath);
end 





