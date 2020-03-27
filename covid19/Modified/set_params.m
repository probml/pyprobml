function params=set_params()
% values from Table 1 of Science paper

beta = 1.12;
mu = 0.55;
Z = 3.69;
D = 3.48;
alpha = 0.14;
theta = 1.36;
params = [beta,mu,theta,Z,alpha,D];
params = params(:);

end
