function params = checkbound_params(params, first_iter)
% params is 6 x num_ens
if nargin < 2, first_iter = false; end

[paramin, paramax] = param_bounds;
nparams = length(paramin);
if first_iter
    for i=1:nparams
        temp=params(i,:);
        index=(temp<paramin(i))|(temp>paramax(i));
        index_out=find(index>0); % out of bounds
        index_in=find(index==0); % in bounds
        %redistribute out bound ensemble members
        if ~isempty(index_out)
         params(i,index_out)=datasample(params(i,index_in),length(index_out));
        end
    end
else
    for i=1:nparams
        ndx = params(i,:) < paramin(i);
        params(i,ndx)=paramin(i)*(1+0.1*rand(sum(ndx),1));
        ndx = params(i,:) > paramax(i);
        params(i,ndx)=paramax(i)*(1-0.1*rand(sum(ndx),1));
    end
   
end

end
