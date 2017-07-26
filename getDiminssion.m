function dim = getDiminssion(X, pram)


if pram == 1
    dim = size(X,1);
else
    dim = cell({[num2str(size(X,1)) 'x' num2str(size(X,2))]});
end
        


end