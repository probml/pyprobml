

P = dlmread('mote_locs.txt'); % id,x,y coords
X = P(:,2:end);
N = size(X,1);
ndx = [1:4, 6:N];  % sensor 5 is faulty, remove it
y = dlmread('lab_temperature_3.txt');
X = X(ndx,:);
y = y(ndx);
save('moteData.mat', 'X','y');