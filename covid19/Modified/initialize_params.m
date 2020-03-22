function [x,paramax,paramin]=initialize_params(num_ens)
% x is (6, num_ens) where the 6 dimensions refer to 
% beta,mu,theta,Z,alpha,D

[paramin, paramax] = param_bounds;
x=lhsu(paramin, paramax, num_ens);
x=x';

end