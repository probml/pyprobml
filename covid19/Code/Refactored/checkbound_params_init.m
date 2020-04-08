function params = checkbound_params_init(params)
% x is 6 x num_ens

[paramin, paramax] = param_bounds;

for i=1:6
    temp=params(i,:);
    index=(temp<paramin(i))|(temp>paramax(i));
    index_out=find(index>0); % out of bounds
    index_in=find(index==0); % in bounds
    %redistribute out bound ensemble members
    params(i,index_out)=datasample(params(i,index_in),length(index_out));
end

end

%{
function x = checkbound_ini(x,pop)
...
for i=1:6
    temp=x(end-6+i,:);
    index=(temp<xmin(i))|(temp>xmax(i));
    index_out=find(index>0);
    index_in=find(index==0);
    %redistribute out bound ensemble members
    x(end-6+i,index_out)=datasample(x(end-6+i,index_in),length(index_out));
end
%}