load('Data/M.mat') %load mobility
load('Data/pop.mat') %load population
load('Data/incidence.mat') %load observation

seed = 42;
num_ens = 3;


pop0=pop*ones(1,num_ens);


rng(seed);
[x0,paramax,paramin]=initialize(M, pop, num_ens); %ok
% x has size (375*5 + 6, num_ens) = (1875 + 6, 2) = 1881, 2
%S,E,Is,Ia,obs,...,beta,mu,theta,Z,alpha,D

t=1;
rng(seed);
[x1,pop1]=SEIR(x0,M,pop,t,pop0);
 
rng(seed);
[x1new,pop1new]=SEIR_new(x0,M,pop,t,pop0);
assert(approxeq(x1, x1new))
assert(approxeq(pop1, pop1new))