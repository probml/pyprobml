load('../../Data/M.mat')
load('../../Data/pop.mat')
load('../../Data/incidence.mat')
Iter = 2;
num_ens = 20;
seeds = 1:1;
for seedi = 1:length(seeds)
    seed = seeds(seedi);
    [para_post,theta] = inference2(M, pop, incidence, Iter, num_ens);
end
