function [beta, mu, theta, Z, alpha, D] = unpack_params(params)

betaidx=1;
muidx=2;
thetaidx=3;
Zidx=4;
alphaidx=5;
Didx=6;

beta=params(betaidx,:);
mu=params(muidx,:);
theta=params(thetaidx,:);
Z=params(Zidx,:);
alpha=params(alphaidx,:);
D=params(Didx,:);
end
