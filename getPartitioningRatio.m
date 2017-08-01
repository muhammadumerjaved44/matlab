function [holdoutRatio, validationRatio]= getPartitioningRatio(trainRatio, validRatio, testRatio)

holdoutRatio = testRatio/100;

train = 100 - testRatio;

% valid = validRatio/train;

validationRatio = validRatio/train;

validationRatio = str2num(sprintf('%.2f',validationRatio));


end