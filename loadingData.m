function [features, classLables, classNames] = loadingData(dataFolder,folderNameC1, folderNameC2)
%% Load the Data from Directory
% Input: set the folder name 
% dataFolder = 'dataSet';
% folderNameC1 = 'catagory1';
% folderNameC2 = 'catagory-1';

% Ouptus:
% features: return the Feature Set in "Instances x Features"
% classLables : returnt the class lables
% classNames : return the unique name for the classes


%% Data Partition 

gcsDatabase = imageSet(dataFolder,'recursive');


featureCount = 0;
for i=1:size(gcsDatabase,2)
    for j = 1:gcsDatabase(i).Count
        featureCount = featureCount + 1;
        currentimage = double((read(gcsDatabase(i),j)));
        features(featureCount,:) = reshape(currentimage,size(currentimage,1)*size(currentimage,2),1);
        classLables{featureCount} = gcsDatabase(i).Description;    
    end
    classNames{i} = gcsDatabase(i).Description;
end

end 