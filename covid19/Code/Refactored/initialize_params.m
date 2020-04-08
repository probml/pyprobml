function [x,paramax,paramin]=initialize_params(num_ens, random)
% x is (6, num_ens) where the 6 dimensions refer to 
% beta,mu,theta,Z,alpha,D

if nargin < 1, num_ens = 1; end
if nargin < 2, random = true; end

[paramin, paramax] = param_bounds;
x=lhsu(paramin, paramax, num_ens, random);
x=x';

end