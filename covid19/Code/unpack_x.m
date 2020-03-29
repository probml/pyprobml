function [states, params] = unpack_x(x)
% x has size (375*5 + 6, num_ens) = (1875 + 6, 2) = 1881, 2
%S,E,Is,Ia,obs,...,beta,mu,theta,Z,alpha,D

[paramin, paramax] = param_bounds();
nparams = length(paramin);
assert(nparams == 6)
[nrows, nens] = size(x);
nloc = (nrows - nparams)/5;
assert(nloc == 375)
ndx = nloc*5;
states = x(1:ndx, :);
params = x(ndx+1:nrows, :);

end
