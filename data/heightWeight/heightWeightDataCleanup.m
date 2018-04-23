% Process the data from http://www.stat.psu.edu/resources/bydata.htm
% requires matlab 7
% Save first 2 features (height, weight) in csv format.
% Delete data which is missing values (3 cases)

fid = fopen('biometric_data.txt');
rawdata = textscan(fid,'%d%n%n%n%n%n%n%n%n',...
		   'headerlines', 1,...
		   'delimiter','\t',...
		   'treatAsEmpty',{'*'});
fclose(fid);

data.C = rawdata{1}+1; % 1=male, 2=female
data.X = [rawdata{2} rawdata{3}];  % height, weight


bad = [];
for d=1:2
  bad = [bad find(isnan(data.X(:,d)))];
end
data.C(bad) = [];
data.X(bad,:) = [];

D = [data.C data.X];
dlmwrite('biometric_data_simple.txt', D)

