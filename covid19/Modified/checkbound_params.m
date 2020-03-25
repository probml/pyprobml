function params = checkbound_params(params)
% params is 6 x num_ens

[paramin, paramax] = param_bounds;
nparams = length(paramin);

for i=1:nparams
    ndx = params(i,:) < paramin(i);
    params(i,ndx)=paramin(i)*(1+0.1*rand(sum(ndx),1));
    ndx = params(i,:) > paramax(i);
    params(i,ndx)=paramax(i)*(1-0.1*rand(sum(ndx),1));
end

end
