load('Data/M.mat') %load mobility
load('Data/pop.mat') %load population
load('Data/incidence.mat') %load observation

seed = 42;
rng(seed);

num_ens = 3;
[x0,paramax,paramin]=initialize(pop, num_ens, M);
% x has size (375*5 + 6, num_ens) = (1875 + 6, 2) = 1881, 2
%S,E,Is,Ia,obs,...,beta,mu,theta,Z,alpha,D
xstates = x0(1:1875,:);
xparams = x0(1876:1881,:);


rng(seed);
[states0]=initialize_state(pop, num_ens, M); % (1875, 2)
[params0, paramax2, paramin2]=initialize_params(num_ens);

assert(approxeq(xparams, params0))
assert(approxeq(xstates, states0))
