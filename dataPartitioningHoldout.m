function [partitionData, info, indices]= dataPartitioningHoldout(dataSet,ratio)
% Input:
% Give the data set as input for partitioning
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
% indices:        
% return the indices of validation set/testing Set form the taining data 
% and use in calculation of class performance 



X = dataSet.X;
Y = dataSet.Y;
cv = cvpartition(Y, 'HoldOut', ratio);
info.holdoutRatio = ratio;
info.NumTestSets= cv.NumTestSets;
info.TrainSize= cv.TrainSize;
info.TestSize= cv.TestSize;
info.Type = cv.Type;
partitionData.cv = cv;

% seperate the data in to teest/validation and training set
partitionData.X1input = X(training(cv), :);
partitionData.Y1output = Y(training(cv), :);
indices = test(cv);
partitionData.X2input = X(test(cv), :);
partitionData.Y2output = Y(test(cv), :);


end