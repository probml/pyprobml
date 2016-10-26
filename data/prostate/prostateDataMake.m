% from http://www-stat.stanford.edu/~tibs/ElemStatLearn/
% See their book p47
fid = fopen('prostate.txt');
C = textscan(fid,'%f%f%f%f%f%f%f%f%f%f%s');
fclose(fid);
% rownum lcavol lweight age lbph svi lcp gleason pgg45 lpsa 
names = {'lcavol', 'lweight', 'age',  'lbph', 'svi', 'lcp', 'gleason', 'pgg45', 'lpsa'};
istrain = [C{end}{:}]=='T'
X = [C{2} C{3} C{4} C{5} C{6} C{7} C{8} C{9}];
y = double([C{10}]);
Xtrain = X(find(istrain),:);
ytrain = y(find(istrain),:);
Xtest = X(find(~istrain),:);
ytest = y(find(~istrain),:);
save('prostate.mat', 'Xtrain', 'ytrain', 'Xtest', 'ytest', 'X','y', 'names','istrain');

X = standardize(X); % standardize outside of train/test split 
Xtrain = X(find(istrain),:);
ytrain = y(find(istrain),:);
Xtest = X(find(~istrain),:);
ytest = y(find(~istrain),:);
save('prostateStnd.mat', 'Xtrain', 'ytrain', 'Xtest', 'ytest', 'X','y', 'names','istrain');
