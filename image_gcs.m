clear all; clc; close all;
%% Creating forlders and Path setting

% you only have to set the folder name or use the same name
dataFolder = 'dataSet';
folderNameC1 = '0';
folderNameC2 = '1';

% check directory if not created than creat it

if (~isdir(dataFolder))
    mkdir(dataFolder);
end

cd(dataFolder);
if (~isdir(folderNameC1))
    mkdir(folderNameC1);
end
if (~isdir(folderNameC2))
    mkdir(folderNameC2);
end
cd ..

% creating your directory Path to save images according to class labes
folder1 = [cd '\' dataFolder '\' folderNameC1 '\'];
folder2 = [cd '\' dataFolder '\' folderNameC2 '\'];
addpath(folder1);
addpath(folder2);
% set the number of images
numOfImage = 500;

%% Image Creation
%Background and foreground object size
backSize = 50; 
foreSize = 15;

% set the limit that object remain in the background
Max = backSize-foreSize;

% intensities level for foregraound and background
backIntensity = 18;
foreIntensity = 210-backIntensity;

% Foregraond and Background size
Ib = uint8(backIntensity*ones(backSize,backSize));
If = uint8(foreIntensity*ones(foreSize,foreSize));

figure;
imshow(Ib)

figure;
imshow(If)

%% Overlaped Images
% randomely set the forgrodund position
rng('shuffle');
Points = randi([0 Max],[numOfImage,2]);

% Resultant image initialize
I = Ib;

% genrate images for other class 1
for i = 1 : length(Points)
        startPoint = Points(i,:);
        I1 = Ib;
        I1((1:size(If,1))+startPoint(1),(1:size(If,2))+startPoint(2),:) = backIntensity+If;
%         figure;
%         imshow(I)
        newimagename = [folder1 num2str(i) 'c1.jpeg'];
        imwrite(I1,newimagename)
end 

% genrate images for other class -1
for j = 1 : length(Points)
        rng('shuffle');
%         randomIntensity = randi([18]);
%         randomIntensity = 192;
        randomIntensity = randi([30 255]);
        I2 = Ib+randomIntensity;
%         I((1:size(If,1))+startPoint(1),(1:size(If,2))+startPoint(2),:) = backIntensity+If;
%         figure;
%         imshow(I)
        newimagename = [folder2 num2str(j) 'c-1.jpeg'];
        imwrite(I2,newimagename)
end 

% testSVD();

% for j = 1 : length(Points)/2
%         rng('shuffle');
% %         randomIntensity = randi([18]);
%         randomIntensity = 210;
%         I2 = Ib+randomIntensity;
% %         I((1:size(If,1))+startPoint(1),(1:size(If,2))+startPoint(2),:) = backIntensity+If;
% %         figure;
% %         imshow(I)
%         newimagename = [folder2 num2str(j) 'c-1.jpeg'];
%         imwrite(I2,newimagename)
% end 


 

