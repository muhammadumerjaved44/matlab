close all; clc; clear all;

%set path for the image folder
myDir1 = 'C:\Users\215-IV\Desktop\projects\gcs\catagory1\';
myDir2 = 'C:\Users\215-IV\Desktop\projects\gcs\catagory-1\';
outputfolder = 'C:\Users\215-IV\Desktop\projects\gcs\SVDfaces\';

testimgpath = 'C:\Users\215-IV\Desktop\projects\gcs\catagory1\20c1.jpeg';

%%
ext_img = '*.jpeg';
a = dir([myDir1 ext_img]);
nfile = max(size(a)) ; % number of image files
% for i=1:nfile
%   my_img(i).img = imread([myDir1 num2str(i) 'c1.jpeg']);
%   files(i).img = [[myDir1 num2str(i) 'c1.jpeg']];
%   labels(i).img = 1;
% end
% 
% b = dir([myDir2 ext_img]);
% nfile = max(size(b)) ; % number of image files
% for i=1:nfile
%   my_img(i+nfile).img = imread([myDir2 num2str(i) 'c-1.jpeg']);
%   files(i+nfile).img = [[myDir2 num2str(i) 'c-1.jpeg']];
%   labels(i+nfile).img = -1;
% end

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

%% folder 2 loading
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

%%
X = X'
[U D V] = svd(X);
redudim = 25;

figure;
bar(diag(D));

xhat = U(:,1:redudim)*D(1:redudim,1:redudim)*V(:,1:redudim)';
%%
for i = 1 : size(xhat,1)
%         eigngcs = [];
        eigngcs = reshape(xhat(i,:),50,50);
        eigngcs = uint8(eigngcs);
%         figure;
%         imshow(eigngcs)
        outpath = [outputfolder 'eigen_gcs_' num2str(i) '.jpeg'];
        imwrite(eigngcs,outpath);
end


%%
Timage = imread(testimgpath);
InImage = reshape(Timage,size(Timage,1)*size(Timage,2),1);
InImage = double(InImage); % Centered test image
[U1 D1 V1] = svd(InImage');
xtest = U1*D1*V1';
%%
    Euc_dist = [];
    for i = 1 : 50
        q = xhat(i,:);
        temp2 = ( norm( xtest - q ) )^2;
        Euc_dist = [Euc_dist temp2];
    end

    [Euc_dist_min , Recognized_index] = min(Euc_dist);
    OutputName = strcat(int2str(Recognized_index),'.jpeg');
    figure;
    
    subplot(1,2,1);
    imshow(Timage,[])
    
    outpath2 = [outputfolder 'eigen_gcs_' OutputName];
    recoImage = imread(outpath2);
    subplot(1,2,2);
    imshow(recoImage,[]);
