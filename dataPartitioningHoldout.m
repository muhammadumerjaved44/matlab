function [partitionData, info, indices]= dataPartitioningHoldout(dataSet,ratio)
X = dataSet.X;
Y = dataSet.Y;
cv = cvpartition(Y, 'HoldOut', ratio);
info.holdoutRatio = ratio;
info.NumObservations= cv.NumObservations;
info.NumTestSets= cv.NumTestSets;
info.TrainSize= cv.TrainSize;
info.TestSize= cv.TestSize;
info.Type = cv.Type;

partitionData.cv = cv;
partitionData.X1input = X(training(cv), :);
partitionData.Y1output = Y(training(cv), :);
indices = test(cv);
partitionData.X2input = X(test(cv), :);
partitionData.Y2output = Y(test(cv), :);


end