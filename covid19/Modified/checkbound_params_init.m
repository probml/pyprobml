function params = checkbound_params_init(params)
% x is 6 x num_ens

[paramin, paramax] = param_bounds;

for i=1:6
    temp=params(i,:);
    index=(temp<paramin(i))|(temp>paramax(i));
    index_out=(index>0); % out of bounds
    index_in=(index==0); % in bounds
    %redistribute out bound ensemble members
    params(i,index_out)=datasample(params(i,index_in),length(index_out));
end

end