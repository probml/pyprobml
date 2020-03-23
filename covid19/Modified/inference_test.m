load('Data/M.mat') %load mobility
load('Data/pop.mat') %load population
load('Data/incidence.mat') %load observation

num_ens = 4;
num_iter = 2;
num_times = 3;
seed = 42;


Td=9;%average reporting delay
a=1.85;%shape parameter of gamma distribution
b=Td/a;%scale parameter of gamma distribution
%gam_rnds=ceil(gamrnd(a,b,1e4,1));%pre-generage gamma random numbers
gam_rnds = [];

rng(seed); 
xpost1 = inference1(M, pop, incidence, num_ens, num_iter, num_times, gam_rnds);
% xpost is 1881 x num_ens x num_times x num_iter
% where 1881 = 1875 + 6, 1875 = 375*5

rng(seed); 
xpost2 = inference2(M, pop, incidence, num_ens, num_iter, num_times, gam_rnds);

assert(approxeq(xpost1, xpost2))